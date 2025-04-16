import numpy as np
import torch
from gymnasium import spaces

from env_models.mec_environment import MECEnvironment
from ppo.ppo_utils.multi_discrete import MultiDiscrete


class PPOEnv(MECEnvironment):
    def __init__(self, configs):
        super(PPOEnv, self).__init__(configs)
        self.num_agents=1
        self.ppo_config = configs['ppo_offloading_config']
        self.ppo_mec_config = configs['ppo_tx_config']
        self.mec_episode_length = self.ppo_mec_config.offloading_episode_length
        self.mec_power_decision = torch.zeros(self.user_task_shape,device=self.device)
        self.mec_offloading_ratio = torch.zeros((self.tot_users * self.n_sbs, self.max_task_deadline + 1),device=self.device)
        self.reward_history = torch.zeros((self.mec_episode_length + self.max_task_deadline, self.tot_users, 1),device=self.device)
        self.reward_counter = 0
        self.time_to_ra=True
        self.continuous_power=self.ppo_config.continuous_tx_power
        self.random_client_selection=self.ppo_config.random_client_selection
        self.random_subchannel=self.ppo_config.random_subchannel
        self.action_space=[]
        action_mask_list=[]
        action_mask = np.zeros((self.num_agents, 1,self.max_users),dtype=np.int64)
        action_mask[:, :,:self.tot_users] = 1
        if not self.random_subchannel:
            sc_allocation_actions = MultiDiscrete([[0, self.num_sc - 1] for _ in range(self.max_users)])
            self.action_space.append(sc_allocation_actions)
            action_mask_list.append(action_mask.copy())

        if self.ppo_config.continuous_mec:
            mec_action_space = spaces.Box(0.0, 1.0, shape=(self.max_users,), dtype=np.float32)
        else:
            mec_actions = [[0, self.ppo_mec_config.split_quantization] for _ in range(self.n_sbs)]
            mec_action_space = MultiDiscrete(mec_actions)
            self.action_space.append(mec_action_space)
            action_mask_list.append(action_mask.copy())
        if self.continuous_power:
            power_action_space = spaces.Box(0.0, 1.0, shape=(self.max_users,), dtype=np.float32)
        else:
            power_action_space = MultiDiscrete([[0, self.env_config.power_quant-self.env_config.quantization_buffer] for _ in range(self.max_users)])
        self.action_space.append(power_action_space)
        action_mask_list.append(action_mask.copy())

        self.action_mask=np.concatenate(action_mask_list,axis=1)
        self.default_action_mask=action_mask
        obs_dim = self.local_tx_observation.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
        state_dim=self.global_state.shape[1]
        self.state_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(state_dim,), dtype=np.float32)

    # def perform_task_offloading_decision(self):
    #     # Assume self.mec_arrival_buffer and self.real_user_mask are already torch tensors.
    #     mec_buffer = self.mec_arrival_buffer[self.real_user_mask == 1].clone()
    #     dones = self.done_tx_tasks.flatten()
    #     to_be_offloaded = (mec_buffer * dones).reshape(-1, 1)
    #     # Assume self.cur_tx_idx is a torch tensor.
    #     cur_tx_idx = self.cur_tx_idx.repeat_interleave(self.n_sbs, dim=0).reshape(-1, 1)
    #     # np.take_along_axis --> torch.gather (here we assume self.mec_offloading_ratio is 2D)
    #     mec_ratio = torch.gather(self.mec_offloading_ratio, 1, cur_tx_idx).reshape(self.tot_users, -1)
    #     mec_portion = mec_ratio * to_be_offloaded
    #     # For flatten('F') we transpose then flatten (column‑major flatten)
    #     flatten_ratio = mec_portion.transpose(0, 1).reshape(-1, 1)
    #     cur_tx_idx = self.cur_tx_idx.t().repeat_interleave(self.n_sbs, dim=0).reshape(-1, 1)
    #     # The following line replicates:
    #     # self.task_mec_size[np.indices(self.task_mec_size.shape)[0], cur_tx_idx] += flatten_ratio
    #     # We assume self.task_mec_size is a 2D torch tensor.
    #     row_indices = torch.arange(self.task_mec_size.size(0), device=self.device) \
    #         .unsqueeze(1).expand(self.task_mec_size.size(0), self.task_mec_size.size(1))
    #     # Use index_put_ with accumulate=True to add values at the specified indices.
    #     self.task_mec_size.index_put_(
    #         (row_indices, cur_tx_idx.squeeze(-1)),
    #         flatten_ratio.squeeze(-1),
    #         accumulate=True
    #     )
    #     mec_buffer[dones == 1] = 0
    #     mec_offload = flatten_ratio.reshape(self.n_sbs, -1)
    #     self.handle_mec_queue(mec_offload)
    #     self.mec_offloading_counter += torch.sum(mec_offload, dim=1)
    #     self.mec_arrival_buffer[self.real_user_mask == 1] = mec_buffer


    def reset(self, seed=None, options=None):
        self.reset_data()
        super().reset(seed,options)
        info={}
        return self.local_tx_observation,self.global_state,self.action_mask,info

    @property
    def avail_split_actions(self):
        split_action_masking = torch.ones(
            (self.tot_users, self.ppo_mec_config.split_quantization + 1),
            dtype=torch.bool,
            device=self.device
        )
        task_arrival_mask = self.ue_arrival_buffer != 0
        avail_split_actions = task_arrival_mask.reshape(-1, 1) == split_action_masking
        cur_loc_time = torch.ceil(
            self.task_loc_size * self.task_power / (self.slot_duration * self.ue_computing_power)
        )
        tot_loc_time = torch.sum(cur_loc_time, dim=1, keepdim=True)
        loc_buffer_extended = self.ue_arrival_buffer.reshape(-1, 1).expand(
            self.tot_users, self.ppo_mec_config.split_quantization + 1
        )
        quantization_extended = (
                                    torch.arange(self.ppo_mec_config.split_quantization + 1, device=self.device,
                                                 dtype=torch.float32)
                                    .unsqueeze(0)
                                    .expand(self.tot_users, -1)
                                ) / self.ppo_mec_config.split_quantization
        loc_size = loc_buffer_extended * quantization_extended
        req_proc_time = torch.ceil(loc_size * self.task_power / self.ue_computing_power / self.slot_duration)
        done_time = tot_loc_time + req_proc_time
        enough_time_mask = done_time <= (self.max_task_deadline * self.max_task_deadline)
        avail_split_actions = avail_split_actions & enough_time_mask
        return avail_split_actions

    @property
    def global_state(self):
        state = self.local_tx_observation.reshape((1, -1))
        return state

    @property
    def local_tx_observation(self):
        cur_second = int(self.cur_timeslot // self.slots_per_coherence)
        ready_to_fl_transmit = (self.fl_train_last_slot < self.cur_timeslot) * self.fl_users * (
                self.fl_remained_data > 0)
        ready_to_transmit = torch.bitwise_or(ready_to_fl_transmit.int(), self.bg_users.int())
        not_ready = ready_to_transmit == 0
        users_gains = self.normalized_gain[:, cur_second % self.channel_buffer_size].clone()
        cur_users_gains = 10 * torch.log10(users_gains)
        # m_gain=cur_users_gains.mean()
        min_gain = torch.min(cur_users_gains)
        users_gains = (cur_users_gains - min_gain) / (torch.max(cur_users_gains) - min_gain)
        users_gains[not_ready] = -1
        obs = users_gains
        obs = torch.concat((obs, self.user_max_tx_power))

        obs = torch.concat((obs, ready_to_fl_transmit), dim=0)
        sc_user = self.user_sc_assignment.clone().flatten().float()
        sc_user[sc_user >= 0] /= (self.tot_users - 1)
        obs = torch.concat((obs, sc_user), dim=0)
        # users_m_gains = torch.zeros(self.max_users).to(self.device)
        if not self.random_client_selection:
            norm_power = torch.zeros(self.max_users).to(self.device)
            training = 1
            if not self.training:
                # mean_gains = self.mean_n_gains_db(self.ppo_offloading_config.gain_n_history)
                # cur_users_m_gain = mean_gains[:self.tot_users]
                # tot_m_gain = cur_users_m_gain.mean()
                # users_m_gains = (mean_gains - tot_m_gain) / (torch.max(cur_users_m_gain) - torch.min(cur_users_m_gain))
                # users_m_gains[self.tot_users:] = 0
                cur_users_power = self.user_power[:self.tot_users]
                min_power = torch.min(cur_users_power)
                max_power = torch.max(cur_users_power)
                norm_power = (self.user_power - min_power) / (max_power - min_power)
                norm_power[self.tot_users:] = -1
                training = 0
            obs = torch.cat((obs, norm_power), dim=0)
            obs = torch.concat((obs, self.fl_mean_grad_norm * (1 - training)))
            obs = torch.concat((obs, self.fl_latest_part * (1 - training) / (self.episode_counter + 1e-6)))
        obs=obs.reshape(1, -1)
        obs=torch.repeat_interleave(obs,self.num_agents,dim=0)
        return obs.reshape(self.num_agents,-1).cpu().numpy()

    @property
    def local_computing_observation(self):
        obs = self.ue_arrival_buffer.reshape(-1, 1)
        obs = torch.cat((obs, self.task_type_buffer.reshape(-1, 1)), dim=1)
        user_rates = self.user_rates_real.reshape(-1, 1)
        cur_tx_time = torch.ceil(self.task_tx_size / (self.slot_duration * user_rates))
        max_val = self.max_task_deadline * self.max_task_deadline
        cur_tx_time[cur_tx_time > max_val] = max_val
        tot_tx_time = torch.sum(cur_tx_time, dim=1, keepdim=True)
        cur_loc_time = torch.ceil(self.task_loc_size * self.task_power / (self.slot_duration * self.ue_computing_power))
        tot_loc_time = torch.sum(cur_loc_time, dim=1, keepdim=True)
        mec_queue = torch.sum((self.mec_queue_idx >= 0), dim=1).reshape(1, -1)
        mec_queue = mec_queue.repeat(self.tot_users, 1)
        obs = torch.cat((obs, tot_loc_time, tot_tx_time, mec_queue), dim=1)
        obs = torch.cat((obs, (self.ue_computing_power / self.ue_config.max_computing_power).reshape(-1, 1)), dim=1)
        mean_mec_active = torch.mean(self.mec_active_history, dim=1).reshape(1, -1).repeat(self.tot_users, 1)
        mec_computing_power = (self.mec_computing_power / self.bs_config.max_computing_power).reshape(1, -1).repeat(
            self.tot_users, 1)
        obs = torch.cat((obs, mec_computing_power), dim=1)
        obs = torch.cat((obs, mean_mec_active), dim=1)
        ue_ids = torch.arange(self.tot_users, device=self.device).reshape(-1, 1)
        obs = torch.cat((obs, ue_ids), dim=1)
        if self.time_to_ra:
            a=2

        return obs


    def step(self, action):
        action=action[0]
        if self.random_subchannel:
            raise NotImplemented('Random SC not implemented!')
        # if self.random_client_selection:
        #     raise NotImplemented('Random Client selection not implemented!')
        if not self.continuous_power:
            raise NotImplemented('Discrete power not implemented!')
        sc_alloc = action[:self.tot_users].flatten().int()
        capacity_counter = np.zeros(self.num_sc, dtype=int)
        # tmp_assignment = np.ones((self.num_sc, self.sc_capacity)) * -1
        self.user_sc_assignment.fill_(-1)
        for ue_idx, sc_idx in enumerate(sc_alloc):
            self.user_sc_assignment[sc_idx, capacity_counter[sc_idx]] = ue_idx
            capacity_counter[sc_idx] += 1
        # if not self.training:
        #     self.fl_users = action[self.max_users:self.max_users*2].flatten().int()
        #     self.fl_latest_part[self.fl_users] = self.episode_counter
        if self.random_client_selection:
            power_action=action[self.max_users:]
            power_action = torch.clamp(power_action, 0.01, 1.0)
            power_action *= torch.as_tensor(self.action_mask[0, 1]).to(self.device)
        else:
            power_action = action[self.max_users*2:]
            power_action = torch.clamp(power_action, 0.01, 1.0)
            power_action *= torch.as_tensor(self.action_mask[0, 2]).to(self.device)

        self.chosen_power_ratio = power_action.float()

        self.perform_noma_step()
        noma_metrics = self.metrics_tx
        # fl_metrics = self.perform_fl_step()
        fl_metrics = None
        # if self.random_subchannel:
        #     random_assignment = self.rng.permutation(self.max_users).reshape(
        #         (self.num_sc, self.sc_capacity))
        #     random_assignment[random_assignment >= self.tot_users] = -1
        #     self.user_sc_assignment = torch.as_tensor(random_assignment).to(self.device)
        # if not self.client_selection and self.ppo_fl_config.random_participation and not self.training:
        #     self.fl_users.zero_()
        #     num_users = self.rng.integers(1, self.tot_users)
        #     chosen_idx = torch.randperm(self.tot_users)[:num_users]
        #     self.fl_users[chosen_idx] = 1
        self.cur_timeslot += 1
        self.prepare_next_noma_step()
        next_obs = self.local_tx_observation
        next_state=self.global_state
        fl_step_metrics=(self.latest_test_acc,self.latest_test_acc_norm)
        reward = self.tx_reward.item()
        cumulative_reward = None
        if fl_metrics is not None:
            # reward+=self.fl_reward.item()
            self.cumulative_reward += reward
        accuracy_latency=self.accuracy_latency if fl_metrics is not None else None

        info = {'noma_metrics': noma_metrics, 'fl_metrics': fl_metrics,'fl_step_metrics':fl_step_metrics,'accuracy_latency':accuracy_latency}

        done = reward < 0 or reward > 2
        # done = self.is_done
        if not self.random_client_selection:
            if self.training:
                self.action_mask[:,1]=np.zeros_like(self.action_mask[:,1])
            else:
                self.action_mask[:,1]=self.default_action_mask[:,0].copy()
        if done:
            cumulative_reward = self.cumulative_reward.item()
            next_obs, next_state,_,_=self.reset()
        info['cumulative_reward']=cumulative_reward
        reward=np.array([reward]).reshape((-1,1))
        return next_obs, next_state,self.action_mask,reward, done, info






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

