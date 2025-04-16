import numpy as np
import torch
from gymnasium import spaces
from env_models.mec_environment import MECEnvironment
from maddpg.maddpg_utils.multi_discrete import MultiDiscrete


class MADDPGEnv(MECEnvironment):
    def __init__(self, configs):
        super(MADDPGEnv, self).__init__(configs)

        self.maddpg_device_offloading_config = configs['maddpg_device_offloading_config']
        self.maddpg_device_tx_config = configs['maddpg_device_config']
        self.single_agent=self.maddpg_device_tx_config.single_agent
        self.num_tx_agents = 1 if self.single_agent else self.n_sbs
        self.num_offloading_agents = 1 if self.single_agent else self.tot_users * self.n_sbs
        self.time_to_ra=True
        self.instant_resource_offload=self.maddpg_device_tx_config.instant_resource_offload
        self.joint_decision=self.maddpg_device_tx_config.joint_decision
        self.continuous_tx_power=self.maddpg_device_tx_config.continuous_tx_power
        # ----- Start  Action Specs----- #
        # We define action_multiplier to deal with single agent (maddpg_device) scenario
        tx_sbs_per_agent = self.n_sbs//self.num_tx_agents
        offloading_user_per_agent = self.global_users //self.num_offloading_agents
        action_mask = np.zeros((self.num_tx_agents, tx_sbs_per_agent, self.max_users), dtype=np.int64)
        action_mask[:, :, :self.tot_users] = 1
        action_mask = action_mask.reshape((self.num_tx_agents, -1))
        self.tx_action_mask = action_mask
        self.default_action_mask = action_mask
        # offloading_action_mask = np.ones((self.num_offloading_agents,  offloading_user_per_agent, self.n_sbs), dtype=np.int64)
        self.offloading_action_mask = np.ones((self.num_offloading_agents, self.n_sbs * offloading_user_per_agent),
                                              dtype=np.int64)

        sc_action_space = MultiDiscrete([[0, self.num_sc - 1] for _ in range(self.max_users*tx_sbs_per_agent)])

        power_action_space = spaces.Box(0.0, 1.0, shape=(self.max_users*tx_sbs_per_agent,), dtype=np.float32)

        sc_power_action=[[0, self.num_sc - 1] for _ in range(self.max_users*tx_sbs_per_agent)]
        sc_power_action.extend([[0, self.maddpg_device_tx_config.tx_power_quantization - self.maddpg_device_tx_config.quantization_buffer] for _ in range(self.max_users * tx_sbs_per_agent)])
        sc_power_action_space = MultiDiscrete(sc_power_action)
        # Offloading decision consists the split ratio and offloading ratio, we omit the last sbs because we can
        # determine the ration based on the current ratios
        offloading_action_space = MultiDiscrete([[0, self.maddpg_device_offloading_config.split_quantization] for _ in range(self.n_sbs*offloading_user_per_agent)])
        joint_offloading_action_space = MultiDiscrete([[0, self.maddpg_device_offloading_config.split_quantization] for _ in range(self.n_sbs*self.max_users*tx_sbs_per_agent)])
        mec_power_action_space = spaces.Box(0.01, 1.0, shape=(self.n_sbs*offloading_user_per_agent,), dtype=np.float32)
        self.action_spaces=[sc_action_space,power_action_space,offloading_action_space,mec_power_action_space,joint_offloading_action_space,sc_power_action_space]
        # ----- End  Action Specs----- #


    def get_available_action_shape(self,action_type):
        if action_type==1:
            return self.offloading_action_mask.shape[1]
        else:
            return self.tx_action_mask.shape[1:]


    def get_observation_space(self,observation_type):
        obs_dim=self.get_observation(observation_type).shape[1]
        return spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)

    def get_state_space(self,state_type):
        state_dim=self.get_global_state(state_type).shape[1]
        return spaces.Box(low=-np.inf, high=+np.inf, shape=(state_dim,), dtype=np.float32)

    def get_action_space(self,action_type):
        if action_type==0:
            if self.continuous_tx_power:
                return self.action_spaces[:2]
            else:
                return self.action_spaces[5]
        elif action_type==1:
            return self.action_spaces[2:4]
        else:
            return [self.action_spaces[0],self.action_spaces[4],self.action_spaces[1],self.action_spaces[3]]

    def reset(self, seed=None, options=None, step_type=0):
        super().reset(seed, options)
        info = {}
        action_mask = self.tx_action_mask if step_type in (0, 2) else self.offloading_action_mask
        return self.get_observation(step_type), self.get_global_state(step_type), action_mask, info

    def get_post_data(self,step_type=0):
        action_mask = self.tx_action_mask if step_type in (0, 2) else self.offloading_action_mask
        return self.get_observation(step_type),  action_mask

    def get_global_state(self,state_type):
        # return self.get_state(state_type)
        return self.get_observation(state_type).reshape(1,-1)

    def get_observation(self,obs_type):
        # obs_type = 0 ==> resource allocation
        # obs_type = 1 ==> task  offloading
        # obs_type = 2 ==> joint decision
        obs=torch.empty((self.n_sbs,0),device=self.device)
        offloading_obs=torch.empty((self.n_sbs,self.max_users,0),device=self.device)
        if obs_type == 0 or obs_type ==2:
            cur_second = int(self.cur_timeslot // self.slots_per_coherence)
            ready_to_transmit = self.bg_users.int()
            not_ready = ready_to_transmit == 0
            users_gains = self.normalized_gain[:, :, cur_second % self.channel_buffer_size].clone()
            cur_users_gains = 10 * torch.log10(users_gains)
            # m_gain=cur_users_gains.mean()
            min_gain = torch.min(cur_users_gains)
            users_gains = (cur_users_gains - min_gain) / (torch.max(cur_users_gains) - min_gain)
            users_gains[not_ready] = -1
            # obs = users_gains
            obs = torch.concat((obs,users_gains), dim=1)
            if obs_type == 0:
                return obs.reshape((self.num_tx_agents,-1)).cpu().numpy()
        if obs_type == 1 or obs_type == 2:
            offloading_obs = self.ue_arrival_buffer.unsqueeze(-1)
            offloading_obs = torch.cat((offloading_obs, self.task_type_buffer.unsqueeze(-1)), dim=2)
            offloading_obs = torch.cat((offloading_obs, self.max_task_deadline_expanded), dim=2)
            # offloading_obs = torch.cat((offloading_obs, self.task_power_expanded), dim=2)
            user_rates = self.user_rates.unsqueeze(-1)
            offloading_obs = torch.cat((offloading_obs, user_rates / self.max_data_rate), dim=2)
            # loc_processing_time = torch.ceil(
            #     self.task_loc_size * self.ue_config.task_power / self.ue_computing_power.unsqueeze(-1))
            # cur_loc_time_sum = torch.clamp(torch.cumsum(loc_processing_time, dim=-1)[:, :, -1], 0,
            #                                self.max_task_deadline)
            # cur_loc_time_norm = cur_loc_time_sum / self.max_task_deadline
            local_queue_delay = self.compute_local_delay()
            local_processing_delay = torch.ceil((self.ue_arrival_buffer * self.ue_config.task_power) / self.ue_computing_power)
            local_delay = local_queue_delay + local_processing_delay
            offloading_obs = torch.cat((offloading_obs, local_delay.unsqueeze(-1)), dim=2)
            offloading_obs = torch.cat((offloading_obs, (self.ue_computing_power / self.max_ue_computing_power).unsqueeze(-1)), dim=2)

            if obs_type == 1:
                mec_computing_power = (self.mec_computing_power / self.max_mec_computing_power).transpose(0,
                                                                                                          1).unsqueeze(
                    0).expand(self.n_sbs, self.max_users, -1)
                offloading_obs = torch.cat((offloading_obs, mec_computing_power), dim=2)
                ue_ids = torch.arange(self.global_users, device=self.device).reshape(self.n_sbs, self.max_users, 1)
                # offloading_obs = torch.cat((offloading_obs, ue_ids / self.global_users), dim=2)
                offloading_obs = torch.cat((offloading_obs, ue_ids ), dim=2)
            if obs_type == 1:
                return offloading_obs.flatten(start_dim=0, end_dim=1).reshape((self.num_offloading_agents,-1)).cpu().numpy()
        if obs_type == 2:
            obs= torch.cat((obs,offloading_obs.flatten(start_dim=1,end_dim=2),self.mec_computing_power/self.max_mec_computing_power),dim=1)
            return obs.reshape((self.num_tx_agents,-1)).cpu().numpy()

    def get_state(self,obs_type):
        # obs_type = 0 ==> resource allocation
        # obs_type = 1 ==> task  offloading
        # obs_type = 2 ==> joint decision
        obs=torch.empty((self.n_sbs,0),device=self.device)
        offloading_obs=torch.empty((self.n_sbs,self.max_users,0),device=self.device)
        if obs_type == 0 or obs_type ==2:
            cur_second = int(self.cur_timeslot // self.slots_per_coherence)
            ready_to_transmit = self.bg_users.int()
            not_ready = ready_to_transmit == 0
            users_gains = self.normalized_gain[:, :, cur_second % self.channel_buffer_size].clone()
            cur_users_gains = 10 * torch.log10(users_gains)
            # m_gain=cur_users_gains.mean()
            min_gain = torch.min(cur_users_gains)
            users_gains = (cur_users_gains - min_gain) / (torch.max(cur_users_gains) - min_gain)
            users_gains[not_ready] = -1
            # obs = users_gains
            obs = torch.concat((obs,users_gains), dim=1)
            if obs_type == 0:
                obs_flatten=obs.flatten()
                obs_flatten=torch.cat((obs_flatten,self.user_max_tx_power))
                return obs_flatten.reshape((1,-1)).cpu().numpy()
        if obs_type == 1 or obs_type == 2:
            offloading_obs = self.ue_arrival_buffer.unsqueeze(-1)
            offloading_obs = torch.cat((offloading_obs, self.task_type_buffer.unsqueeze(-1)), dim=2)
            # offloading_obs = torch.cat((offloading_obs, self.max_task_deadline_expanded), dim=2)
            # offloading_obs = torch.cat((offloading_obs, self.task_power_expanded), dim=2)
            user_rates = self.user_rates.unsqueeze(-1)
            offloading_obs = torch.cat((offloading_obs, user_rates / self.max_data_rate), dim=2)
            # loc_processing_time = torch.ceil(
            #     self.task_loc_size * self.ue_config.task_power / self.ue_computing_power.unsqueeze(-1))
            # cur_loc_time_sum = torch.clamp(torch.cumsum(loc_processing_time, dim=-1)[:, :, -1], 0,
            #                                self.max_task_deadline)
            # cur_loc_time_norm = cur_loc_time_sum / self.max_task_deadline
            local_queue_delay = self.compute_local_delay()
            local_processing_delay = torch.ceil((self.ue_arrival_buffer * self.ue_config.task_power) / self.ue_computing_power)
            local_delay = local_queue_delay + local_processing_delay
            offloading_obs = torch.cat((offloading_obs, local_delay.unsqueeze(-1)), dim=2)
            offloading_obs = torch.cat((offloading_obs, (self.ue_computing_power / self.max_ue_computing_power).unsqueeze(-1)), dim=2)

            # if obs_type == 1:
            #     mec_computing_power = (self.mec_computing_power / self.max_mec_computing_power).transpose(0,
            #                                                                                               1).unsqueeze(
            #         0).expand(self.n_sbs, self.max_users, -1)
            #     offloading_obs = torch.cat((offloading_obs, mec_computing_power), dim=2)
                # ue_ids = torch.arange(self.global_users, device=self.device).reshape(self.n_sbs, self.max_users, 1)
                # offloading_obs = torch.cat((offloading_obs, ue_ids / self.global_users), dim=2)
                # offloading_obs = torch.cat((offloading_obs, ue_ids ), dim=2)
            if obs_type == 1:
                offloading_obs_flatten=offloading_obs.flatten()
                mec_computing_power = (self.mec_computing_power / self.max_mec_computing_power).flatten()
                offloading_obs_flatten=torch.cat((offloading_obs_flatten,mec_computing_power))
                return offloading_obs_flatten.reshape((1,-1)).cpu().numpy()
        if obs_type == 2:
            obs= torch.cat((obs,offloading_obs.flatten(start_dim=1,end_dim=2),self.mec_computing_power/self.max_mec_computing_power),dim=1)
            return obs.reshape((self.num_tx_agents,-1)).cpu().numpy()

    def step(self, action,step_type=0):
        if step_type == 0:
            return self.tx_step(action)
        elif step_type==1:
            return self.offloading_step(action)
        else:
            return self.joint_step(action)

    def joint_step(self,action):
        noma_metrics=None
        # if self.num_tx_agents==1:
        #     action=action.reshape(())
        if self.time_to_ra:
            sc_alloc = action[:,:self.max_users]
            assigned = self.assign_users_to_subchannels(torch.as_tensor(sc_alloc, dtype=torch.long, device=self.device))
            self.user_sc_assignment = assigned
            power_action = action[:, -self.max_users:]
            power_action = torch.clamp(power_action, 0.01, 1.0)
            power_action *= torch.as_tensor(self.tx_action_mask[:, 2]).to(self.device)
            self.chosen_power_ratio = power_action.float()
            self.perform_noma_step()
            noma_metrics = self.metrics_tx
            self.prepare_next_noma_step()
        self.perform_early_steps()
        offloading_actions=action[:, self.max_users:-self.max_users].reshape((self.n_sbs,self.max_users,self.n_sbs))
        split_actions = offloading_actions[:,:,0] / self.maddpg_device_offloading_config.split_quantization
        self.task_split_ratio = torch.as_tensor(split_actions, dtype=self.task_split_ratio.dtype, device=self.device)
        idx = self.cur_timeslot % (self.max_task_deadline + 1)
        offloading_decision = offloading_actions[:,:,1:].reshape(self.n_sbs, self.max_users,
                                                    self.n_sbs - 1) / self.maddpg_device_offloading_config.split_quantization
        self.task_offloading_decision[:, :, :-1, idx] = offloading_decision
        rem_split = 1 - split_actions-offloading_decision.sum(-1)
        self.task_offloading_decision[:, :, -1, idx] = rem_split
        self.perform_later_steps()
        offloading_metrics = self.metrics_offloading
        info = {'noma_metrics':noma_metrics,'offloading_metrics': offloading_metrics}
        reward = self.offloading_reward.cpu().numpy()
        done = reward < -10
        self.time_to_ra = (self.cur_timeslot % self.slots_per_coherence) == 0 or self.instant_resource_offload
        if self.time_to_ra:
            self.tx_action_mask[:, 0,:]=self.default_action_mask.copy()
            self.tx_action_mask[:, 2,:]=self.default_action_mask.copy()
        else:
            self.tx_action_mask[:, 0]=np.zeros_like(self.tx_action_mask[:, 0])
            self.tx_action_mask[:, 2]=np.zeros_like(self.tx_action_mask[:, 2])
        self.tx_action_mask[:,1] = (self.ue_arrival_buffer.sum(-1,keepdim=True) > 0).expand(-1,self.max_users).cpu().numpy()
        return self.get_observation(2), self.tx_action_mask, reward, done, info

    def tx_step(self, action):
        if self.single_agent:
            action = action.reshape((2, self.n_sbs, self.max_users)).transpose(0, 1).flatten(start_dim=1, end_dim=2)
        sc_alloc = action[:, :self.max_users]
        assigned = self.assign_users_to_subchannels(torch.as_tensor(sc_alloc, dtype=torch.long, device=self.device))
        self.user_sc_assignment = assigned
        power_action = action[:, self.max_users:]
        if not self.continuous_tx_power:
            power_action = (power_action + self.maddpg_device_tx_config.quantization_buffer) / self.maddpg_device_tx_config.tx_power_quantization
        power_action = torch.clamp(power_action, 0.01, 1.0)
        power_action *= torch.as_tensor(self.tx_action_mask).to(self.device).reshape((self.n_sbs, self.max_users))
        self.chosen_power_ratio = power_action.float()
        # self.cur_timeslot+=1
        self.perform_noma_step()
        noma_metrics = self.metrics_tx
        self.prepare_next_noma_step()
        info = {'noma_metrics': noma_metrics}
        reward = self.tx_reward.cpu().numpy()
        done = reward < -10
        return self.get_observation(0), self.get_global_state(0), self.tx_action_mask, reward, done, info

    def offloading_step(self, action):
        if self.single_agent:
            action = action.reshape((self.global_users, -1))
        self.perform_early_steps()
        split_actions = action[:, 0].reshape(
            (self.n_sbs, self.max_users)) / self.maddpg_device_offloading_config.split_quantization
        self.task_split_ratio = torch.as_tensor(split_actions, dtype=self.task_split_ratio.dtype, device=self.device)
        idx = self.cur_timeslot % (self.max_task_deadline + 1)
        offloading_decision = action[:, 1:self.n_sbs].reshape(self.n_sbs, self.max_users,
                                                              self.n_sbs - 1) / self.maddpg_device_offloading_config.split_quantization
        self.task_offloading_decision[:, :, :-1, idx] = offloading_decision
        rem_split = 1 - split_actions - offloading_decision.sum(-1)
        self.task_offloading_decision[:, :, -1, idx] = rem_split
        mec_power_ratio = action[:, self.n_sbs:].reshape(self.n_sbs, self.max_users, self.n_sbs)
        self.task_mec_power_ratio[:, :, :, idx] = mec_power_ratio
        # self.task_split_ratio = self.task_split_ratio.zero_() + 0.1
        # self.task_offloading_decision[:, :, 0, :] = 1
        # self.task_offloading_decision[:, :, 1, :] = 0.5
        # self.task_offloading_decision[:, :, 2, :] = 0.25
        self.perform_later_steps()
        offloading_metrics = self.metrics_offloading
        info = {'offloading_metrics': offloading_metrics}
        reward = self.offloading_reward.cpu().numpy()
        done = reward < -10
        self.offloading_action_mask = (self.ue_arrival_buffer.reshape((self.global_users, 1)) > 0).expand(-1,
                                                                                                          self.n_sbs).reshape(
            (self.num_offloading_agents, -1)).cpu().numpy()
        return self.get_observation(1), self.get_global_state(1), self.offloading_action_mask, reward, done, info



