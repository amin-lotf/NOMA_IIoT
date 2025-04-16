import gymnasium as gym
import numpy as np
import torch
from numpy.core.defchararray import upper


class BaseEnvironment(gym.Env):
    def __init__(self, configs):
        self.env_name = configs['env_name']
        self.glob_env_name = configs['glob_env_name']
        self.trained_env = configs['trained_env']
        self.device = configs['device']
        self.tpdv = dict(device=self.device, dtype=torch.float32)
        self.env_config = configs['env_config']
        self.bs_config = configs['bs_config']
        self.ue_config = configs['ue_config']
        # self.task_config = configs['task_config']
        self.rng = configs['rng']
        self.n_sbs = self.env_config.n_sbs
        self.num_sc = self.bs_config.num_sc
        self.sc_capacity = self.bs_config.sc_capacity
        self.user_max_tx_power = torch.tensor(np.round((10 ** (self.ue_config.tx_power / 10) / 1000), 4)).unsqueeze(
            0).to(**self.tpdv)
        self.tolerable_power = torch.tensor(np.round((10 ** (self.bs_config.tolerable_power / 10) / 1000), 4)).unsqueeze(
            0).to(**self.tpdv)
        self.max_users = self.num_sc * self.sc_capacity
        self.global_users = self.max_users * self.n_sbs
        # self.tot_users = self.bs_config.tot_users
        self.tot_users = self.max_users
        self.tot_user_mask = torch.repeat_interleave(
            torch.arange(self.max_users).reshape((1, self.max_users)).to(self.device), self.n_sbs, dim=0).reshape(
            self.n_sbs, self.max_users) >= self.tot_users
        random_assignment = self.rng.permutation(self.max_users).reshape(
            (1, self.num_sc, self.sc_capacity))
        random_assignment = np.repeat(random_assignment, self.n_sbs, axis=0)
        random_assignment[random_assignment >= self.tot_users] = -1
        self.user_sc_assignment = torch.as_tensor(random_assignment).to(self.device)
        self.channel_buffer_size = self.env_config.channel_buffer_size
        self.cur_timeslot = 0
        self.slots_per_coherence = int(self.env_config.coherence_time / self.env_config.slot_duration)
        self.slot_duration = self.env_config.slot_duration
        self.bg_users = torch.ones((self.n_sbs, self.max_users), dtype=torch.short).to(self.device)
        self.bg_users[self.tot_user_mask] = 0
        self.ue_computing_power = torch.rand(size=(self.n_sbs, self.max_users), device=self.device)*(self.ue_config.max_computing_power-self.ue_config.min_computing_power)+self.ue_config.min_computing_power
        self.mec_computing_power = torch.rand(size=(self.n_sbs, 1),device=self.device)*(self.bs_config.max_computing_power-self.bs_config.min_computing_power)+self.bs_config.min_computing_power
        self.ue_computing_power[self.tot_user_mask] = 0
        self.task_power = self.ue_config.task_power
        self.metadata = {'n_sc': self.num_sc, 'sc_capacity': self.sc_capacity}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
