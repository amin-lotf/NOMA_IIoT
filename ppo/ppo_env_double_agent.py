import numpy as np
import torch
from gymnasium import spaces

from env_models.mec_environment import MECEnvironment
from ppo.ppo_utils.multi_discrete import MultiDiscrete


class PPOEnv(MECEnvironment):
    def __init__(self, configs):
        super(PPOEnv, self).__init__(configs)
        self.ppo_fl_config = configs['ppo_fl_config']
        self.ppo_config = configs['ppo_offloading_config']
        self.continuous_power=self.ppo_config.continuous_tx_power
        self.random_sc=self.ppo_config.random_subchannel
        #====== NOMA ======#
        if self.continuous_power:
            if self.random_sc:
                self.action_space = spaces.Box(0.00,1.0,shape=(self.max_users,),dtype=np.float32)
            else:
                sc_allocation_actions = MultiDiscrete([[0, self.num_sc - 1] for _ in range(self.max_users)])
                tx_power_action_space = spaces.Box(0.00,1.0,shape=(self.max_users,),dtype=np.float32)
                self.action_space=[sc_allocation_actions,tx_power_action_space]
            self.action_mask = torch.zeros(1, self.max_users).to(self.device).int()
            self.action_mask[0, 0:self.tot_users] = 1
        else:
            if self.random_sc:
                raise NotImplemented('Multi discrete with random SC not implemented!')
            alpha_actions = [[0, self.env_config.power_quant - 1] for _ in range(self.max_users)]
            sc_allocation_actions = [[0, self.num_sc - 1] for _ in range(self.max_users)]
            sc_allocation_actions.extend(alpha_actions)
            self.action_space = MultiDiscrete(sc_allocation_actions)
            self.action_mask = torch.zeros(2, self.max_users).to(self.device).int()
            self.action_mask[:, :self.tot_users] = 1
            self.action_mask=self.action_mask.reshape((1,-1))
        obs_dim = self.local_tx_observation.shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
        # ====== FL ======#
        self.client_selection=self.ppo_config.client_selection
        if self.client_selection:
            if self.ppo_fl_config.continuous_action:
                self.fl_action_space = spaces.Box(0.0, 1.0, shape=(self.max_users,), dtype=np.float32)
            else:
                client_actions = [[0, 1] for _ in range(self.max_users)]
                self.fl_action_space = MultiDiscrete(client_actions)
            fl_obs_dim = self.fl_observation.shape[0]
            self.fl_observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(fl_obs_dim,), dtype=np.float32)
            self.fl_action_mask = torch.zeros(1, self.max_users).to(self.device).int()
            self.fl_action_mask[0, 0:self.tot_users] = 1

    @property
    def local_tx_observation(self):
        cur_second = int(self.cur_timeslot // self.slots_per_coherence)
        ready_to_fl_transmit = (self.fl_train_last_slot < self.cur_timeslot) * self.fl_users * (
                    self.fl_remained_data > 0)
        ready_to_transmit = torch.bitwise_or(ready_to_fl_transmit.int(), self.bg_users.int())
        not_ready = ready_to_transmit==0
        users_gains = self.normalized_gain[:, cur_second % self.channel_buffer_size].clone()
        cur_users_gains=users_gains[:self.tot_users]
        m_gain=cur_users_gains.mean()
        users_gains=(users_gains-m_gain)/(torch.max(cur_users_gains) - torch.min(cur_users_gains))
        users_gains[not_ready] = -1
        obs = users_gains
        obs = torch.concat((obs, self.user_max_tx_power))

        obs = torch.concat((obs, ready_to_fl_transmit), dim=0)
        sc_user = self.user_sc_assignment.clone().flatten().float()
        sc_user[sc_user>=0]/=(self.tot_users-1)
        obs = torch.concat((obs, sc_user), dim=0)
        return obs.flatten()

    @property
    def fl_observation(self):
        mean_gains = self.mean_n_gains_db(self.ppo_fl_config.gain_n_history)
        cur_users_m_gain=mean_gains[:self.tot_users]
        tot_m_gain = cur_users_m_gain.mean()
        users_m_gains = (mean_gains - tot_m_gain) / (torch.max(cur_users_m_gain) - torch.min(cur_users_m_gain))
        users_m_gains[self.tot_users:]= -1
        obs=users_m_gains
        cur_users_m_gain = self.user_power[:self.tot_users]
        mean_power=cur_users_m_gain.mean()
        norm_power=(self.user_power-mean_power)/(torch.max(cur_users_m_gain)-torch.min(cur_users_m_gain))
        norm_power[self.tot_users:]= -1
        obs = torch.concat((obs,  norm_power))
        obs=torch.concat((obs,self.fl_mean_grad_norm))
        obs=torch.concat((obs,self.fl_latest_part/(self.episode_counter+1e-6)))
        # obs = torch.concat((obs,  self.episode_counter.unsqueeze(0)))
        return obs.flatten()



    def step(self, action):
        if self.random_sc:
            if self.continuous_power:
                action = torch.clamp(action, 0.01, 1.0)
                action*=self.action_mask[0,:self.max_users]
                self.chosen_power_ratio=action[0]
            else:
                raise NotImplemented('Multi discrete with random SC not implemented!')
        else:
            sc_alloc = action[0, :self.tot_users].flatten().int()
            capacity_counter = np.zeros(self.num_sc, dtype=int)
            # tmp_assignment = np.ones((self.num_sc, self.sc_capacity)) * -1
            self.user_sc_assignment.fill_(-1)
            for ue_idx, sc_idx in enumerate(sc_alloc):
                self.user_sc_assignment[sc_idx, capacity_counter[sc_idx]] = ue_idx
                capacity_counter[sc_idx] += 1
            if self.continuous_power:
                power_action = torch.clamp(action[0, self.max_users:], 0.01, 1.0)
                power_action*=self.action_mask[0,:self.max_users]
                self.chosen_power_ratio = power_action
            else:
                self.cur_power_idx = action[0, self.tot_users:].reshape((-1, 1)) + 1
        self.perform_noma_step()
        noma_metrics = self.metrics_tx
        fl_metrics = self.perform_fl_step()
        if self.random_sc:
            random_assignment = self.rng.permutation(self.max_users).reshape(
                (self.num_sc, self.sc_capacity))
            random_assignment[random_assignment >= self.tot_users] = -1
            self.user_sc_assignment = torch.as_tensor(random_assignment).to(self.device)
        # if not self.client_selection and self.ppo_fl_config.random_participation and not self.training:
        #     self.fl_users.zero_()
        #     num_users = self.rng.integers(1, self.tot_users)
        #     chosen_idx = torch.randperm(self.tot_users)[:num_users]
        #     self.fl_users[chosen_idx] = 1
        self.cur_timeslot += 1
        self.prepare_next_noma_step()
        next_obs = self.local_tx_observation
        fl_step_metrics=(self.latest_test_acc,self.latest_test_acc_norm)
        info = {'noma_metrics': noma_metrics, 'fl_metrics': fl_metrics,'fl_step_metrics':fl_step_metrics}
        reward = self.tx_reward
        done = torch.logical_or(reward < 0, reward > 2)
        return next_obs, reward, done, self.action_mask, info

    def step_fl(self, action):
        if self.ppo_fl_config.continuous_action:
            action=torch.clamp(action,0.0,1.0)
            action*=self.fl_action_mask
            binary_selection = torch.distributions.Binomial(1, action.flatten()).sample()
            self.fl_users = binary_selection.int()
        else:
            self.fl_users=action.flatten().int()
        self.fl_latest_part[self.fl_users]=self.episode_counter

    @property
    def fl_info(self):
        cumulative_reward=None
        reward = self.computing_reward
        self.cumulative_reward+=reward[0][0]
        done=self.is_done

        if done:
            cumulative_reward = self.cumulative_reward.item()
            self.reset_model()
        info={}
        next_obs = self.fl_observation
        return next_obs,reward,done,self.fl_action_mask,cumulative_reward,info



    # @property
    # def is_done(self):
    #     n = self.task_config.test_history_buffer // 2
    #     indices_last_n = torch.arange(self.episode_counter, self.episode_counter - n, -1).to(
    #         self.device) % self.task_config.test_history_buffer
    #
    #     # Get indices for the previous A elements
    #     indices_prev_n = torch.arange(self.episode_counter - n, self.episode_counter - 2 * n, -1).to(
    #         self.device) % self.task_config.test_history_buffer
    #
    #     # Gather the values from ACC
    #     last_n_values = self.avg_test_acc[indices_last_n]
    #     prev_n_values = self.avg_test_acc[indices_prev_n]
    #
    #     # Compute means
    #     mean_last_n = torch.mean(last_n_values)
    #     mean_prev_n = torch.mean(prev_n_values)
    #
    #     # Compute the difference
    #     diff_means = mean_last_n - mean_prev_n
    #     done = diff_means < self.task_config.accuracy_threshold if not self.env_config.testing_env else False
    #     return done

