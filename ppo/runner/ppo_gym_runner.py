import time

import numpy as np
import torch
from matplotlib.style.core import available
from torch.utils.tensorboard import SummaryWriter

from general_utils import OffloadingPerformanceTracker, TxPerformanceTracker, gens
from general_utils.noma_server import transmit_and_wait
from ppo.runner.base_ppo_gym_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class AlgoRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, env):
        super(AlgoRunner, self).__init__(env)

    def run(self):
        tx_episodes = int(self.num_env_steps) // self.tx_episode_length
        # We assume slot duration is less than a second
        glob_step=0
        tx_episode=0
        tx_step=0
        tx_step_type=  0
        total_rewards=[]
        self.warmup(is_offloading=False)
        while glob_step <= self.num_env_steps:
            if self.use_tx_linear_lr_decay:
                self.tx_trainer.policy.lr_decay(tx_episode, tx_episodes)
            values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(
                tx_step, is_offloading=False)
            next_obs, next_state, avail_actions, rewards, dones, info = self.envs.step(actions.cpu().numpy(),step_type=tx_step_type)
            # tx_reward = tx_rewards[0]
            reward = info['done_reward']
            if reward is not None:
                total_rewards.append(reward)
                m_reward = np.mean(total_rewards[-100:])
                print(
                    f":{glob_step} done {len(total_rewards)} games, reward {m_reward:.3f}")
                self.writer.add_scalar("reward_100", m_reward, glob_step)
                self.writer.add_scalar("reward", reward, glob_step)
            data = next_obs, next_state, rewards, dones, avail_actions, values, actions.cpu().numpy(), action_log_probs, rnn_states, rnn_states_critic
            # insert data into offloading_buffer
            self.insert(data,is_offloading=False)
            tx_step+=1
            if tx_step==self.tx_episode_length:
                tx_step=0
                self.compute(is_offloading=False)
                train_infos = self.train(is_offloading=False)
                # self.writer.add_scalar("tx_value_loss", train_infos['value_loss'], glob_step)
                # self.writer.add_scalar("tx_policy_loss", train_infos['policy_loss'], glob_step)
                # self.writer.add_scalar("tx_dist_entropy", train_infos['dist_entropy'], glob_step)
                # self.writer.add_scalar("tx_actor_grad_norm", train_infos['actor_grad_norm'],
                #                        glob_step)
                # self.writer.add_scalar("tx_critic_grad_norm", train_infos['critic_grad_norm'],
                #                        glob_step)
                # self.writer.add_scalar("tx_ratio", train_infos['ratio'], glob_step)
                tx_episode+=1
            glob_step+=1
        print('End of the simulation!')

    def warmup(self,is_offloading=False):
        step_type= 0
        obs,state,available_action,_ = self.envs.reset(step_type=step_type)
        buffer =  self.tx_buffer
        buffer.obs[0] = torch.as_tensor(obs).to(self.device)
        buffer.state[0] = torch.as_tensor(state).to(self.device)
        buffer.available_actions[0] = torch.as_tensor(available_action).to(self.device)


    @torch.no_grad()
    def collect(self, step,is_offloading=False):
        trainer =  self.tx_trainer
        buffer =  self.tx_buffer
        n_agents =  self.num_tx_agents
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
        buffer =  self.tx_buffer
        obs,next_state, rewards, dones, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        masks = dones == False
        buffer.insert(obs, next_state, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards,
                                      masks, available_actions=available_actions)

