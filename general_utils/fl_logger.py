import numpy as np
from scipy import io

from general_utils import gens
from general_utils.gens import get_results_directory


class OffloadingPerformanceTracker:
    def __init__(self, env, config,fast_results=False):
        self.env = env
        self.config = config
        self.save_vals = False
        self.epoch_save_period = self.config.save_period if self.env.env_config.testing_env else 1000
        self.step_save_period = self.config.save_period if self.env.env_config.testing_env else 1000
        self.choose_every = 1  # choose every  nth element of array
        self.collect_after = 1  # collect data after ... days
        self.collect_latest = 10000000
        self.mean_latest = 1000
        self.skip_first=100 if self.env.env_config.testing_env else 0
        self.epoch_limit = 600001
        self.step_limit = 600001

    def __enter__(self):
        self.fl_ep_counter = 0
        self.full_train_counter = 0
        self.step_counter = 0
        self.step_bits_balance = np.zeros((self.epoch_limit,))
        self.step_drop_ratio = np.zeros((self.epoch_limit,))
        self.step_step_bits_balance = np.zeros((self.step_limit,))
        self.step_step_drop_ratio = np.zeros((self.step_limit,))
        self.step_ep = np.zeros((self.epoch_limit,))
        self.step_delay = np.zeros((self.epoch_limit,))
        self.step_power = np.zeros((self.epoch_limit,))
        self.step_bits_done = np.zeros((self.epoch_limit,))
        self.step_sensitivity_delay = np.zeros((self.epoch_limit,))
        self.step_mean_power = np.zeros((self.epoch_limit,))
        self.step_mean_split = np.zeros((self.epoch_limit,))
        self.step_reward = np.zeros((self.epoch_limit,))
        self.cumulative_reward = np.zeros((self.epoch_limit,))
        self.accuracy_latency = None
        self.latest_acc_idx = 0
        # setting the counters to 1 for saving purpose
        self.fl_ep_counter = 1
        self.full_train_counter = 1
        self.step_counter = 1
        self.save_data()
        self.save_step_data()
        self.fl_ep_counter = 0
        self.full_train_counter = 0
        self.step_counter = 0
        return self

    def __exit__(self, *args):
        # self.writer.close()
        pass

    def record_performance(self, metrics, step_reward, last_step=False):
        step_ep, step_delay, step_power, step_bits_done, step_bits_balance, step_drop_ratio,mean_sensitivity_delay,mean_power,mean_split = metrics
        self.step_ep[self.fl_ep_counter] = step_ep
        self.step_delay[self.fl_ep_counter] = step_delay
        self.step_power[self.fl_ep_counter] = step_power
        self.step_bits_done[self.fl_ep_counter] = step_bits_done
        self.step_bits_balance[self.fl_ep_counter] = step_bits_balance
        self.step_drop_ratio[self.fl_ep_counter] = step_drop_ratio
        self.step_sensitivity_delay[self.fl_ep_counter] = mean_sensitivity_delay
        self.step_mean_power[self.fl_ep_counter] = mean_power
        self.step_mean_split[self.fl_ep_counter] = mean_split
        self.step_reward[self.fl_ep_counter] = step_reward
        # self.accuracy_latency=accuracy_latency.copy()
        # self.latest_acc_idx=int(latest_acc_idx)
        # if cumulative_reward is not None:
        #     self.cumulative_reward[self.full_train_counter]=cumulative_reward
        #     self.full_train_counter+=1
        self.fl_ep_counter += 1

        if last_step or (self.fl_ep_counter > self.collect_after and self.fl_ep_counter % self.epoch_save_period == 0):
            self.save_data()

        if self.fl_ep_counter >= self.epoch_limit:
            print(f'tot collected:{self.epoch_limit / self.choose_every}')

    def save_data(self):
        mean_ep = self.step_ep[:self.fl_ep_counter].mean()
        self.save_vals = True
        res_dir = get_results_directory(f'{gens.get_project_root()}/{self.config.results_dir}')
        io.savemat(f'{res_dir}/{self.env.env_name}_offloading_reward.mat',
                   {f'{self.env.env_name}_offloading_reward': self.step_reward[self.skip_first:self.fl_ep_counter:self.choose_every]})

        io.savemat(
            f'{res_dir}/{self.env.env_name}_step_bits_balance.mat',
            {f'{self.env.env_name}_step_bits_balance': self.step_bits_balance[self.skip_first:self.fl_ep_counter:self.choose_every]})
        io.savemat(
            f'{res_dir}/{self.env.env_name}_step_drop_ratio.mat',
            {f'{self.env.env_name}_step_drop_ratio': self.step_drop_ratio[
                                                     self.skip_first:self.fl_ep_counter:self.choose_every]})
        # io.savemat(
        #     f'{gens.get_results_directory(self.config.results_dir)}/{self.env.env_name}_step_bits_balance_norm_m.mat',
        #     {f'{self.env.env_name}_step_bits_balance_norm_m': np.mean(self.step_drop_ratio[
        #                                                      mean_idx:self.fl_ep_counter])})
        # 
        io.savemat(
            f'{res_dir}/{self.env.env_name}_step_delay.mat',
            {f'{self.env.env_name}_step_delay': self.step_delay[self.skip_first:self.fl_ep_counter:self.choose_every]})

        io.savemat(
            f'{res_dir}/{self.env.env_name}_step_offloading_power.mat',
            {f'{self.env.env_name}_step_power': self.step_power[self.skip_first:self.fl_ep_counter:self.choose_every]})

        io.savemat(
            f'{res_dir}/{self.env.env_name}_step_bits_done.mat',
            {f'{self.env.env_name}_step_bits_done': self.step_bits_done[self.skip_first:self.fl_ep_counter:self.choose_every]})

        io.savemat(
            f'{res_dir}/{self.env.env_name}_step_ep.mat',
            {f'{self.env.env_name}_step_ep': self.step_ep[self.skip_first:self.fl_ep_counter:self.choose_every]})

        io.savemat(
            f'{res_dir}/{self.env.env_name}_step_sensitivity_delay.mat',
            {f'{self.env.env_name}_step_sensitivity_delay': self.step_sensitivity_delay[self.skip_first:self.fl_ep_counter:self.choose_every]})
        io.savemat(
            f'{res_dir}/{self.env.env_name}_step_mean_power.mat',
            {f'{self.env.env_name}_step_mean_power': self.step_mean_power[self.skip_first:self.fl_ep_counter:self.choose_every]})
        io.savemat(
            f'{res_dir}/{self.env.env_name}_step_mean_split.mat',
            {f'{self.env.env_name}_step_mean_split': self.step_mean_split[self.skip_first:self.fl_ep_counter:self.choose_every]})
        #
        print()
        print(
            f'{self.env.env_name}, steps {self.fl_ep_counter}, mean ep: {mean_ep:.2f}')

    def record_step_performance(self, metrics):
        acc_test, acc_test_norm = metrics
        self.step_step_bits_balance[self.step_counter] = acc_test
        self.step_step_drop_ratio[self.step_counter] = acc_test_norm

        if self.step_counter > self.skip_first + 1 and self.step_counter % self.step_save_period == 0:
            self.save_step_data()
        self.step_counter += 1

    def save_step_data(self):
        start_idx = max(self.step_counter - self.collect_latest, self.skip_first)
        res_dir = get_results_directory(f'{gens.get_project_root()}/{self.config.results_dir}')
        io.savemat(
            f'{res_dir}/{self.env.env_name}_step_step_bits_balance.mat',
            {f'{self.env.env_name}_step_step_bits_balance': self.step_step_bits_balance[
                                                            start_idx:self.step_counter:self.choose_every]})
        io.savemat(
            f'{res_dir}/{self.env.env_name}_step_step_bits_balance_norm.mat',
            {f'{self.env.env_name}_step_step_bits_balance_norm': self.step_step_drop_ratio[
                                                                 start_idx:self.step_counter:self.choose_every]})
