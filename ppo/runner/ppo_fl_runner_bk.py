import time

import numpy as np
import torch
from matplotlib.style.core import available
from torch.utils.tensorboard import SummaryWriter

from general_utils import OffloadingPerformanceTracker, TxPerformanceTracker, gens
from general_utils.noma_server import transmit_and_wait
from ppo.runner.base_ppo_fl_runner import Runner



def _t2n(x):
    return x.detach().cpu().numpy()


class AlgoRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, env):
        super(AlgoRunner, self).__init__(env)

    def run(self, logging_config):
        episodes = int(self.num_env_steps) // self.offloading_episode_length
        # We assume slot duration is less than a second
        slot_per_seconds=1/self.env.slot_duration
        glob_step=0
        tx_episode=0
        offloading_episode=0
        tx_step=0
        offloading_step=0
        with (OffloadingPerformanceTracker(self.env, logging_config) as offloading_tracker):
            with TxPerformanceTracker(self.env, logging_config) as tx_tracker:
                self.warmup(is_offloading=False)
                if not self.joint_decision:
                    self.warmup(is_offloading=True)
                while glob_step <= episodes:
                    if glob_step%slot_per_seconds==0:
                        if self.use_tx_linear_lr_decay:
                            self.tx_trainer.policy.lr_decay(tx_episode, episodes)
                        self.handle_post_data(is_offloading=False)
                        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(
                            tx_step, is_offloading=False)
                        next_obs, next_state, avail_actions, rewards, dones, info = self.envs.step(actions,step_type=0)
                        # tx_reward = tx_rewards[0]
                        noma_metrics = info['noma_metrics']
                        if noma_metrics is not None:
                            tx_tracker.record_performance(noma_metrics, rewards[0])
                        data = next_obs, next_state, rewards, dones, avail_actions, values, actions.cpu().numpy(), action_log_probs, rnn_states, rnn_states_critic
                        # insert data into offloading_buffer
                        self.insert(data,is_offloading=False)
                        tx_step+=1
                        if tx_step==self.tx_episode_length:
                            tx_step=0
                            self.compute(is_offloading=False)
                            train_infos = self.train(is_offloading=False)
                            tx_episode+=1
                            if tx_episode % self.tx_save_interval == 0 or tx_episode == episodes - 1:
                                self.save_model(is_offloading=False)
                                train_infos["average_episode_rewards"] = np.mean(
                                    self.tx_buffer.rewards) * self.tx_episode_length
                                # computing_tracker.record_ppo_metrics(train_infos["policy_loss"], train_infos["value_loss"],
                                #                                      train_infos["average_episode_rewards"])
                                print(
                                   f" Time slot: {self.env.cur_timeslot}, average Tx episode rewards: {train_infos['average_episode_rewards']}")

                    if self.use_offloading_linear_lr_decay:
                        self.offloading_trainer.policy.lr_decay(offloading_episode, episodes)
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(
                        offloading_step, is_offloading=True)

                    next_obs, next_state, avail_actions, rewards, dones, info = self.envs.step(actions, step_type=1)
                    # tx_reward = tx_rewards[0]
                    offloading_metrics = info['offloading_metrics']
                    if offloading_metrics is not None:
                        offloading_tracker.record_performance(offloading_metrics, rewards[0])

                    data = next_obs, next_state, rewards, dones, avail_actions, values, actions.cpu().numpy(), action_log_probs, rnn_states, rnn_states_critic
                    # insert data into offloading_buffer
                    self.insert(data, is_offloading=True)
                    offloading_step += 1
                    if offloading_step == self.offloading_episode_length:
                        offloading_step = 0
                        self.compute(is_offloading=True)
                        train_infos = self.train(is_offloading=True)
                        offloading_episode += 1
                        if offloading_episode % self.offloading_save_interval == 0 or offloading_episode == episodes - 1:
                            self.save_model(is_offloading=True)
                            train_infos["average_episode_rewards"] = np.mean(
                                self.offloading_buffer.rewards) * self.offloading_episode_length
                            # computing_tracker.record_ppo_metrics(train_infos["policy_loss"], train_infos["value_loss"],
                            #                                      train_infos["average_episode_rewards"])
                            print(
                                f" Time slot: {self.env.cur_timeslot}, average Offloading episode rewards: {train_infos['average_episode_rewards']}")
                    glob_step+=1

    def warmup(self,is_offloading=False):
        step_type= 2 if self.joint_decision else is_offloading
        obs,state,available_action,_ = self.envs.reset(step_type=step_type)
        buffer = self.offloading_buffer if is_offloading else self.tx_buffer
        buffer.obs[0] = torch.as_tensor(obs).to(self.device)
        buffer.state[0] = torch.as_tensor(state).to(self.device)
        buffer.available_actions[0] = torch.as_tensor(available_action).to(self.device)

    def handle_post_data(self,is_offloading=False):
        obs,state,available_action = self.envs.get_post_data(step_type=is_offloading)
        buffer = self.offloading_buffer if is_offloading else self.tx_buffer
        buffer.insert_post_data(obs,state,available_action)

    @torch.no_grad()
    def collect(self, step,is_offloading=False):
        trainer = self.offloading_trainer if is_offloading else self.tx_trainer
        buffer = self.offloading_buffer if is_offloading else self.tx_buffer
        n_agents = self.num_offloading_agents if is_offloading else self.num_tx_agents
        trainer.prep_rollout()
        values, actions, action_log_probs, rnn_states, _, rnn_states_critic \
            = trainer.policy.get_actions(self.merge(buffer.obs[step]),
                                                                    self.merge(buffer.state[step]),
                                                                    self.merge(buffer.rnn_states[step]),
                                                                    self.merge(buffer.rnn_states_critic[step]),
                                                                    self.merge(buffer.masks[step]),
                                                                    self.merge(buffer.available_actions[step])
                                                                    )
        values=self.unmerge(values,n_agents)
        actions=self.unmerge(actions,n_agents)
        action_log_probs=self.unmerge(action_log_probs,n_agents)
        rnn_states=self.unmerge(rnn_states,n_agents)
        rnn_states_critic=self.unmerge(rnn_states_critic,n_agents)
        return values, actions, action_log_probs, rnn_states, rnn_states_critic



    def insert(self, data,is_offloading=False):
        buffer = self.offloading_buffer if is_offloading else self.tx_buffer
        obs,next_state, rewards, dones, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        masks = dones == False
        buffer.insert(obs, next_state, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards,
                                      masks, available_actions=available_actions)

