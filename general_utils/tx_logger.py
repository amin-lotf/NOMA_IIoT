import numpy as np
import torch
from scipy import io
from general_utils import gens
from general_utils.gens import get_results_directory


class TxPerformanceTracker:
    def __init__(self, env, config,fast_results=False):
        self.env = env
        self.config = config
        self.save_vals = False
        self.skip_first=100 if self.env.env_config.testing_env else 0
        self.save_period = self.config.save_period if self.env.env_config.testing_env else 1000 if not fast_results else 100
        self.choose_every = 1  # choose every  nth element of array
        self.collect_after = 1  # collect data after ... days
        self.collect_till = 15000000
        self.step_limit = 500000
        self.mean_latest = 100

    def __enter__(self):

        self.step_num_failed_pdsc = np.zeros((self.step_limit,))
        self.step_pc = np.zeros((self.step_limit,))
        self.step_reward = np.zeros((self.step_limit,))
        self.step_m_rate = np.zeros((self.step_limit,))
        self.step_sum_rate = np.zeros((self.step_limit,))
        self.step_done_ratio = np.zeros((self.step_limit,))
        self.step_failed_rate = np.zeros((self.step_limit,))
        self.step_failed_power = np.zeros((self.step_limit,))

        self.actor_loss = np.zeros((self.step_limit,))
        self.critic_loss = np.zeros((self.step_limit,))
        self.episode_reward = np.zeros((self.step_limit,))
        # -------setting the counters to 1 for saving purpose--------
        self.step_counter = 1
        self.episode_counter = 1
        self.save_data()
        self.step_counter = 0
        self.episode_counter = 0
        # ---------------------------------------------
        return self

    def __exit__(self, *args):
        # self.writer.close()
        pass

    def record_performance(self, metrics, step_reward,last_step=False):
        step_m_rate, step_num_failed_pdsc,step_power,step_sum_rate = metrics
        self.step_num_failed_pdsc[self.step_counter] = step_num_failed_pdsc
        self.step_pc[self.step_counter] = step_power
        self.step_reward[self.step_counter] = step_reward
        self.step_m_rate[self.step_counter] = step_m_rate
        self.step_sum_rate[self.step_counter] = step_sum_rate
        # self.step_done_ratio[self.step_counter]=tot_done_ratio
        # self.step_failed_rate[self.step_counter]=sum_failed_rate
        # self.step_failed_power[self.step_counter]=sum_failed_diff_power
        if last_step or (self.step_counter > self.collect_after and self.step_counter % self.save_period == 0):
            self.save_vals = True
            self.save_data()
        self.step_counter += 1
        if self.step_counter >= self.step_limit:
            print(f'tot collected:{self.step_limit / self.choose_every}')
    def save_data(self):
        mean_idx = max(self.step_counter - self.mean_latest, 0)
        mean_rate = self.step_m_rate[:self.step_counter].mean()
        mean_failed_pdsc = self.step_num_failed_pdsc[:self.step_counter].mean()
        res_dir = get_results_directory(f'{gens.get_project_root()}/{self.config.results_dir}')
        io.savemat(
            f'{res_dir}/{self.env.env_name}_step_num_failed_pdsc.mat',
            {f'{self.env.env_name}_step_num_failed_pdsc': self.step_num_failed_pdsc[
                                                          self.skip_first:self.step_counter:self.choose_every]})
        io.savemat(
            f'{res_dir}/{self.env.env_name}_step_num_failed_pdsc_m.mat',
            {f'{self.env.env_name}_step_num_failed_pdsc_m': np.mean(self.step_num_failed_pdsc[
                                                                    mean_idx:self.step_counter])})
        #
        io.savemat(f'{res_dir}/{self.env.env_name}_step_pc.mat',
                   {f'{self.env.env_name}_step_pc': self.step_pc[self.skip_first:self.step_counter:self.choose_every]})
        io.savemat(f'{res_dir}/{self.env.env_name}_step_pc_m.mat',
                   {f'{self.env.env_name}_step_pc_m': np.mean(self.step_pc[mean_idx:self.step_counter])})

        io.savemat(f'{res_dir}/{self.env.env_name}_step_m_rate.mat',
                   {f'{self.env.env_name}_step_m_rate': self.step_m_rate[self.skip_first:self.step_counter:self.choose_every]})
        io.savemat(f'{res_dir}/{self.env.env_name}_step_m_rate_m.mat',
                   {f'{self.env.env_name}_step_m_rate_m': np.mean(self.step_m_rate[mean_idx:self.step_counter])})

        io.savemat(
            f'{res_dir}/{self.env.env_name}_step_sum_rate.mat',
            {f'{self.env.env_name}_step_sum_rate': self.step_sum_rate[self.skip_first:self.step_counter:self.choose_every]})

        io.savemat(f'{res_dir}/{self.env.env_name}_step_reward.mat',
                   {f'{self.env.env_name}_step_reward': self.step_reward[self.skip_first:self.step_counter:self.choose_every]})
        io.savemat(f'{res_dir}/{self.env.env_name}_step_reward_m.mat',
                   {f'{self.env.env_name}_step_reward_m': np.mean(self.step_reward[mean_idx:self.step_counter])})
        print()
        print(
            f'{self.env.env_name}, steps {self.step_counter}, mean rate: {mean_rate:.2f}, mean failed PDSC: {mean_failed_pdsc:0.2f} ')

