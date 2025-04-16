import numpy as np
import torch
from gymnasium import spaces

from duel_ddqn.duel_ddqn_utils.multi_discrete import MultiDiscrete
from env_models.mec_environment import MECEnvironment



class DuelDDQNEnv(MECEnvironment):
    def __init__(self, configs):
        super(DuelDDQNEnv, self).__init__(configs)

        self.duel_ddqn_offloading_config = configs['duel_ddqn_offloading_config']
        self.duel_ddqn_tx_config = configs['duel_ddqn_config']
        self.single_agent=self.duel_ddqn_tx_config.single_agent
        self.num_tx_agents = 1 if self.single_agent else self.n_sbs
        self.num_offloading_agents = 1 if self.single_agent else  self.n_sbs
        self.single_es=self.duel_ddqn_offloading_config.single_es
        # ----- Start  Action Specs----- #
        # We define action_multiplier to deal with single agent (duel_ddqn) scenario
        tx_sbs_per_agent = self.n_sbs//self.num_tx_agents
        offloading_user_per_agent = self.global_users //self.num_offloading_agents
        action_mask = np.zeros((self.num_tx_agents,  self.max_users), dtype=np.int64)
        action_mask[:,  :self.tot_users] = 1
        self.tx_action_mask = action_mask
        self.default_action_mask = action_mask
        # offloading_action_mask = np.ones((self.num_offloading_agents,  offloading_user_per_agent, self.n_sbs), dtype=np.int64)
        self.offloading_action_mask = np.ones((self.num_offloading_agents, offloading_user_per_agent), dtype=np.int64)
        # if self.single_es:
        # self.task_offloading_decision=torch.zeros_like(self.task_offloading_decision)
        # self.task_offloading_decision[self.sbs_range.unsqueeze(-1),self.max_users_range,self.sbs_range.unsqueeze(-1)]=1
        self.offloading_action_mask = np.ones((self.num_offloading_agents, offloading_user_per_agent), dtype=np.int64)
        # else:
        #     self.offloading_action_mask = np.ones((self.num_offloading_agents, self.n_sbs*offloading_user_per_agent), dtype=np.int64)

        # Offloading decision consists the split ratio and offloading ratio, we omit the last sbs because we can
        # determine the ration based on the current ratios
        sc_power_action = [[0, self.num_sc - 1] for _ in range(self.max_users * tx_sbs_per_agent)]
        sc_power_action.extend(
            [[0, self.duel_ddqn_tx_config.tx_power_quantization - self.duel_ddqn_tx_config.quantization_buffer] for _ in
             range(self.max_users * tx_sbs_per_agent)])
        sc_power_action_space = MultiDiscrete(sc_power_action)
        offloading_action=[[0, self.duel_ddqn_offloading_config.split_quantization] for _ in range(offloading_user_per_agent)]
        offloading_action.extend([[0, self.duel_ddqn_offloading_config.mec_power_quantization] for _ in range(offloading_user_per_agent)])
        offloading_action_space = MultiDiscrete(offloading_action)
        self.action_spaces=[sc_power_action_space,offloading_action_space]
        # ----- End  Action Specs----- #


    def get_available_action_shape(self,action_type):
        if action_type==1:
            return self.offloading_action_mask.shape[1]
        else:
            return self.tx_action_mask.shape[1:]


    def get_observation_space(self,observation_type):
        obs_dim=self.get_observation(observation_type).shape[1]
        return spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)



    def get_action_space(self,action_type):
        if action_type==0:
            return self.action_spaces[0]
        elif action_type==1:
            return self.action_spaces[1]
        else:
            raise NotImplemented('action not supported!')

    def reset(self, seed=None, options=None,step_type=0):
        super().reset(seed,options)
        info={}
        action_mask=self.tx_action_mask if step_type in (0,2) else self.offloading_action_mask
        return self.get_observation(step_type),action_mask,info

    def get_post_data(self, step_type=0):
        action_mask = self.tx_action_mask if step_type in (0, 2) else self.offloading_action_mask
        return self.get_observation(step_type), action_mask

    def get_observation(self, obs_type):
        # obs_type = 0 ==> resource allocation
        # obs_type = 1 ==> task  offloading
        # obs_type = 2 ==> joint decision
        obs = torch.empty((self.n_sbs, 0), device=self.device)
        offloading_obs = torch.empty((self.n_sbs, self.max_users, 0), device=self.device)
        if obs_type == 0:
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
            obs = torch.concat((obs, users_gains, self.user_max_tx_power.reshape((1, -1)).expand(self.n_sbs, -1)),
                               dim=1)
            return obs.reshape((self.num_tx_agents, -1)).cpu().numpy()
        if obs_type == 1:
            offloading_obs = self.ue_arrival_buffer
            offloading_obs = torch.cat((offloading_obs, self.task_type_buffer), dim=1)
            # offloading_obs = torch.cat((offloading_obs, self.max_task_deadline_expanded.squeeze()), dim=1)
            offloading_obs = torch.cat((offloading_obs, self.ue_computing_power / self.max_ue_computing_power), dim=1)
            offloading_obs = torch.cat((offloading_obs, self.user_rates / self.max_data_rate), dim=1)
            # loc_processing_time = torch.ceil(
            #     self.task_loc_size * self.ue_config.task_power / self.ue_computing_power.unsqueeze(-1))
            # cur_loc_time_sum = torch.clamp(torch.cumsum(loc_processing_time, dim=-1)[:, :, -1], 0,
            #                                self.max_task_deadline)
            # cur_loc_time_norm = cur_loc_time_sum / self.max_task_deadline
            local_queue_delay = self.compute_local_delay()
            local_processing_delay = torch.ceil(
                (self.ue_arrival_buffer * self.ue_config.task_power) / self.ue_computing_power)
            local_delay = local_queue_delay + local_processing_delay
            offloading_obs = torch.cat((offloading_obs, local_delay), dim=1)
            mec_computing_power = (self.mec_computing_power / self.max_mec_computing_power)
            offloading_obs = torch.cat((offloading_obs, mec_computing_power), dim=1)
            # ue_ids = torch.arange(self.global_users, device=self.device).reshape(self.n_sbs, self.max_users)
            # # offloading_obs = torch.cat((offloading_obs, ue_ids / self.global_users), dim=2)
            # offloading_obs = torch.cat((offloading_obs, ue_ids ), dim=1)
            return offloading_obs.cpu().numpy()
        if obs_type == 2:
            obs = torch.cat((obs, offloading_obs.flatten(start_dim=1, end_dim=2),
                             self.mec_computing_power / self.max_mec_computing_power), dim=1)
            return obs.reshape((self.num_tx_agents, -1)).cpu().numpy()

    def step(self, action,step_type=0):
        if step_type == 0:
            return self.tx_step(action)
        elif step_type==1:
            return self.offloading_step(action)
        else:
            raise NotImplemented('Action not supported!')



    def tx_step(self,action):
        sc_alloc = action[:,:self.max_users]
        assigned=self.assign_users_to_subchannels(torch.as_tensor(sc_alloc, dtype=torch.long,device=self.device))
        self.user_sc_assignment=assigned
        power_action = action[:,self.max_users:]
        power_action = (power_action + self.duel_ddqn_tx_config.quantization_buffer) / self.duel_ddqn_tx_config.tx_power_quantization
        # power_action = torch.clamp(power_action, 0.01, 1.0)
        power_action *= torch.as_tensor(self.tx_action_mask).to(self.device)
        self.chosen_power_ratio = power_action.float()
        self.perform_noma_step()
        noma_metrics = self.metrics_tx
        #remove this -----
        # self.cur_timeslot+=1
        self.prepare_next_noma_step()
        info = {'noma_metrics': noma_metrics}
        reward = self.tx_reward.cpu().numpy()
        done = reward < -10
        return self.get_observation(0), self.tx_action_mask,reward,done,info

    def offloading_step(self, action):
        self.perform_early_steps()
        split_actions=action[:,:self.max_users]/self.duel_ddqn_offloading_config.split_quantization
        self.task_split_ratio=torch.as_tensor(split_actions,dtype=self.task_split_ratio.dtype,device=self.device)
        idx = self.cur_timeslot % (self.max_task_deadline + 1)
        self.task_offloading_decision[self.sbs_range.unsqueeze(-1),self.max_users_range,self.sbs_range.unsqueeze(-1),idx]=(1-self.task_split_ratio)*torch.as_tensor(self.offloading_action_mask).to(self.device)
        mec_power_ratio = (action[:, self.max_users:]+self.duel_ddqn_offloading_config.quantization_buffer)/self.duel_ddqn_offloading_config.mec_power_quantization
        self.task_mec_power_ratio[self.sbs_range.unsqueeze(-1),self.max_users_range,self.sbs_range.unsqueeze(-1),idx] = mec_power_ratio
        self.perform_later_steps()
        offloading_metrics = self.metrics_offloading
        info = {'offloading_metrics': offloading_metrics}
        reward = self.offloading_reward.cpu().numpy()
        done = reward < -10
        if self.single_es:
            self.offloading_action_mask=(self.ue_arrival_buffer.reshape((self.global_users,1))>0).reshape((self.num_offloading_agents,-1)).int().cpu().numpy()

        else:
            self.offloading_action_mask=(self.ue_arrival_buffer.reshape((self.global_users,1))>0).expand(-1,self.n_sbs).reshape((self.num_offloading_agents,-1)).int().cpu().numpy()
        return self.get_observation(1),  self.offloading_action_mask, reward, done, info



