import time

import numpy as np
import torch
from general_utils import OffloadingPerformanceTracker, TxPerformanceTracker, gens
from maddpg_device.runners.base_maddpg_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class AlgoRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, env):
        super(AlgoRunner, self).__init__(env)

    def run(self, logging_config):
        # We assume slot duration is less than a second
        slot_per_seconds = 1 / self.env.slot_duration
        glob_step = 0
        tx_step = 0
        offloading_step = 0
        tx_step_type = 0
        with (OffloadingPerformanceTracker(self.env, logging_config) as offloading_tracker):
            with TxPerformanceTracker(self.env, logging_config) as tx_tracker:
                tx_obs, tx_state, tx_available_actions = self.warmup(is_offloading=False)
                offloading_obs, offloading_state, offloading_available_actions = self.warmup(is_offloading=True)
                while glob_step <= self.num_env_steps:
                    if  glob_step % slot_per_seconds == 0:
                        self.tx_policy.epsilon_decay(tx_step)
                        # if False:
                        #     if self.use_tx_linear_lr_decay:
                        #         self.tx_trainer.policy.lr_decay(tx_episode, tx_episodes)
                        tx_actions = self.collect(tx_obs, tx_available_actions, is_offloading=False)
                        next_tx_obs, next_tx_state, next_tx_avail_actions, tx_rewards, tx_dones, tx_info = self.envs.step(
                            tx_actions, step_type=tx_step_type)
                        tx_masks = tx_dones == False
                        noma_metrics = tx_info['noma_metrics']
                        if noma_metrics is not None:
                            tx_tracker.record_performance(noma_metrics, tx_rewards[0])
                        data = tx_obs, next_tx_obs, tx_state, next_tx_state, tx_actions, tx_rewards, tx_masks, tx_available_actions, next_tx_avail_actions
                        # insert data into offloading_buffer
                        self.insert(data, is_offloading=False)
                        tx_obs=next_tx_obs
                        tx_state=next_tx_state
                        tx_available_actions=next_tx_avail_actions
                        tx_step += 1
                        if len(self.tx_buffer) >= self.maddpg_tx_config.start_train_step:
                            train_infos = self.train(is_offloading=False)
                            # self.writer.add_scalar("offload_critic_grad_norm", train_infos['critic_grad_norm'], glob_step)
                            # self.writer.add_scalar("offload_ratio", train_infos['ratio'], glob_step)
                            if tx_step % self.tx_save_interval == 0 or tx_step == self.num_env_steps - 1:
                                self.save_model(is_offloading=False)
                                train_infos["average_episode_rewards"] = self.tx_buffer.get_average_rewards(100)
                                # computing_tracker.record_ppo_metrics(train_infos["policy_loss"], train_infos["value_loss"],
                                #                                      train_infos["average_episode_rewards"])
                                print(
                                    f" Time slot: {self.env.cur_timeslot}, average Tx episode rewards: {train_infos['average_episode_rewards']} tx noise:{self.tx_policy.noise_epsilon}")
                    # if False:
                    self.offloading_policy.epsilon_decay(offloading_step)
                    offloading_actions = self.collect(offloading_obs, offloading_available_actions,is_offloading=True)
                    offloading_next_obs, offloading_next_state, offloading_next_avail_actions, offloading_rewards, offloading_dones, offloading_info = self.envs.step(
                        offloading_actions, step_type=1)
                    offloading_masks = offloading_dones == False
                    offloading_metrics = offloading_info['offloading_metrics']
                    if offloading_metrics is not None:
                        offloading_tracker.record_performance(offloading_metrics, offloading_rewards[0])

                    data = offloading_obs, offloading_next_obs, offloading_state, offloading_next_state, offloading_actions, offloading_rewards, offloading_masks, offloading_available_actions, offloading_next_avail_actions
                    self.insert(data, is_offloading=True)
                    offloading_obs=offloading_next_obs
                    offloading_available_actions=offloading_next_avail_actions
                    offloading_state=offloading_next_state
                    offloading_step += 1
                    if len(self.offloading_buffer) >= self.maddpg_offloading_config.start_train_step:
                        train_infos = self.train(is_offloading=True)
                        if offloading_step % self.offloading_save_interval == 0 or offloading_step == glob_step - 1:
                            self.save_model(is_offloading=True)
                            train_infos["average_episode_rewards"] = self.offloading_buffer.get_average_rewards(100)
                            # computing_tracker.record_ppo_metrics(train_infos["policy_loss"], train_infos["value_loss"],
                            #                                      train_infos["average_episode_rewards"])
                            print(
                                f" Time slot: {self.env.cur_timeslot}, average Offloading episode rewards: {train_infos['average_episode_rewards']}, offloading noise: {self.offloading_policy.noise_scale}")
                    glob_step += 1
                print('End of the simulation!')

    def warmup(self, is_offloading=False):
        step_type =  is_offloading
        obs, state, available_action, _ = self.envs.reset(step_type=step_type)
        return obs, state, available_action

    @torch.no_grad()
    def collect(self, obs, available_actions, is_offloading=False):
        trainer = self.offloading_trainer if is_offloading else self.tx_trainer
        n_agents = self.num_offloading_agents if is_offloading else self.num_tx_agents
        trainer.prep_rollout()
        obs_t = torch.as_tensor(obs).to(self.device)
        available_actions_t = torch.as_tensor(available_actions).to(self.device)
        actions = trainer.policy.act(self.merge(obs_t), self.merge(available_actions_t))
        return self.unmerge(actions, n_agents)

    def insert(self, data, is_offloading=False):
        buffer = self.offloading_buffer if is_offloading else self.tx_buffer
        buffer.insert(data)
