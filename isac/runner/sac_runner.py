import time

import numpy as np
import torch
from matplotlib.style.core import available
from torch.utils.tensorboard import SummaryWriter

from general_utils import OffloadingPerformanceTracker, TxPerformanceTracker, gens
from isac.runner.base_runner import Runner



def _t2n(x):
    return x.detach().cpu().numpy()


class AlgoRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, env):
        super(AlgoRunner, self).__init__(env)

    def run(self, logging_config):
        # We assume slot duration is less than a second
        slot_per_seconds=1/self.env.slot_duration
        glob_step=0
        tx_episode=0
        offloading_episode=0
        tx_step=0
        offloading_step=0
        tx_step_type= 2 if self.joint_decision else 0
        with (OffloadingPerformanceTracker(self.env, logging_config) as offloading_tracker):
            with TxPerformanceTracker(self.env, logging_config) as tx_tracker:
                tx_obs,tx_state,tx_available_actions=self.warmup(is_offloading=False)
                if not self.joint_decision:
                    offloading_obs,offloading_state,offloading_available_actions=self.warmup(is_offloading=True)
                while glob_step <= self.num_env_steps:
                    if  self.joint_decision or glob_step%slot_per_seconds==0:
                    # if  False:
                        if self.use_tx_linear_lr_decay:
                            actor_lr,critic_lr=self.tx_trainer.policy.lr_decay(tx_step, self.num_env_steps)
                            self.writer.add_scalar("actor_lr", actor_lr, glob_step)
                            self.writer.add_scalar("critic_lr", critic_lr, glob_step)
                        actions = self.collect(tx_obs,tx_available_actions, is_offloading=False)
                        next_obs,next_state, next_avail_actions, rewards, dones, info = self.envs.step(actions,step_type=tx_step_type)
                        masks = dones == False
                        # tx_reward = tx_rewards[0]
                        noma_metrics = info['noma_metrics']
                        if noma_metrics is not None:
                            tx_tracker.record_performance(noma_metrics, rewards[0])
                        if self.joint_decision:
                            offloading_metrics = info['offloading_metrics']
                            if offloading_metrics is not None:
                                offloading_tracker.record_performance(offloading_metrics, rewards[0])
                        data = tx_obs,tx_state,next_obs,next_state, actions.cpu().numpy(),rewards, masks, tx_available_actions,next_avail_actions
                        # insert data into offloading_buffer
                        self.insert(data,is_offloading=False)
                        tx_available_actions = next_avail_actions
                        tx_obs=next_obs
                        tx_state=next_state
                        if not self.joint_decision:
                            self.handle_post_data(is_offloading=False)
                        tx_step+=1
                        if len(self.tx_buffer) >= self.sac_tx_config.start_train_step:
                            train_infos = self.train(is_offloading=False)
                            # self.writer.add_scalar("critic_loss", train_infos['critic_loss'], glob_step)
                            # self.writer.add_scalar("actor_loss", train_infos['actor_loss'], glob_step)
                            # self.writer.add_scalar("alpha_loss", train_infos['alpha_loss'], glob_step)
                            # self.writer.add_scalar("alpha", train_infos['alpha'], glob_step)
                            # self.writer.add_scalar("offload_critic_grad_norm", train_infos['critic_grad_norm'], glob_step)
                            # self.writer.add_scalar("offload_ratio", train_infos['ratio'], glob_step)
                            if tx_step % self.tx_save_interval == 0 or tx_step == self.num_env_steps - 1:
                                self.save_model(is_offloading=False)
                                train_infos["average_episode_rewards"] = self.tx_buffer.get_average_rewards(100)
                                # computing_tracker.record_ppo_metrics(train_infos["policy_loss"], train_infos["value_loss"],
                                #                                      train_infos["average_episode_rewards"])
                                print(
                                   f" Time slot: {self.env.cur_timeslot}, average Tx episode rewards: {train_infos['average_episode_rewards']}")
                    if not self.joint_decision:
                    # if False:
                        if self.use_offloading_linear_lr_decay:
                            self.offloading_trainer.policy.lr_decay(glob_step, self.num_env_steps)
                        actions = self.collect( offloading_obs,offloading_available_actions, is_offloading=True)

                        next_obs,next_state, next_avail_actions, rewards, dones, info = self.envs.step(actions, step_type=1)
                        masks = dones == False
                        # tx_reward = tx_rewards[0]
                        offloading_metrics = info['offloading_metrics']
                        if offloading_metrics is not None:
                            offloading_tracker.record_performance(offloading_metrics, rewards[0])

                        data = offloading_obs,offloading_state,next_obs,next_state, actions.cpu().numpy(),rewards, masks, offloading_available_actions,next_avail_actions
                        # insert data into offloading_buffer
                        self.insert(data, is_offloading=True)
                        offloading_available_actions=next_avail_actions
                        offloading_obs=next_obs
                        offloading_state=next_state
                        offloading_step += 1
                        if len(self.offloading_buffer) >= self.sac_offloading_config.start_train_step:
                            train_infos = self.train(is_offloading=True)
                            if offloading_step % self.offloading_save_interval == 0 or offloading_step == glob_step - 1:
                                self.save_model(is_offloading=True)
                                train_infos["average_episode_rewards"] = self.offloading_buffer.get_average_rewards(100)
                                # computing_tracker.record_ppo_metrics(train_infos["policy_loss"], train_infos["value_loss"],
                                #                                      train_infos["average_episode_rewards"])
                                print(
                                    f" Time slot: {self.env.cur_timeslot}, average Offloading episode rewards: {train_infos['average_episode_rewards']}")
                    glob_step+=1
                print('End of the simulation!')

    def warmup(self,is_offloading=False):
        step_type= 2 if self.joint_decision else is_offloading
        obs,state,available_action,_ = self.envs.reset(step_type=step_type)
        obs_t = torch.as_tensor(obs).to(self.device)
        state_t=torch.as_tensor(state).to(self.device)
        available_actions_t = torch.as_tensor(available_action).to(self.device)
        return obs_t,state_t,available_actions_t

    def handle_post_data(self,is_offloading=False):
        buffer = self.offloading_buffer if is_offloading else self.tx_buffer
        buffer.handle_post_data()

    @torch.no_grad()
    def collect(self, obs,available_actions,is_offloading=False):
        trainer = self.offloading_trainer if is_offloading else self.tx_trainer
        n_agents = self.num_offloading_agents if is_offloading else self.num_tx_agents
        trainer.prep_rollout()
        obs_t = torch.as_tensor(obs).to(self.device)
        available_actions_t = torch.as_tensor(available_actions).to(self.device)
        actions, _, = trainer.policy.get_actions(self.merge(obs_t),self.merge(available_actions_t))
        return  self.unmerge(actions,n_agents)



    def insert(self, data,is_offloading=False):
        buffer = self.offloading_buffer if is_offloading else self.tx_buffer
        buffer.insert(data)

