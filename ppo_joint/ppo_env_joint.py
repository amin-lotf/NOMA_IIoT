import numpy as np
import torch
from gymnasium import spaces
from env_models.mec_environment import MECEnvironment
from ppo.ppo_utils.multi_discrete import MultiDiscrete


class PPOEnv(MECEnvironment):
    def __init__(self, configs):
        super(PPOEnv, self).__init__(configs)

        self.ppo_config = configs['ppo_joint_config']
        self.single_agent=self.ppo_config.single_agent
        self.num_agents = 1 if self.single_agent else self.n_sbs
        self.time_to_ra=True
        self.instant_resource_offload=self.ppo_config.instant_resource_offload
        self.continuous_tx_power=self.ppo_config.continuous_tx_power
        # ----- Start  Action Specs----- #
        # We define action_multiplier to deal with single agent (PPO) scenario
        action_mask = np.zeros((self.num_agents, 1, self.max_users), dtype=np.int64)
        action_mask[:, :, :self.tot_users] = 1
        self.action_mask = action_mask.repeat(2, axis=1)
        self.default_action_mask = action_mask.squeeze(1)
        sc_action_space = MultiDiscrete([[0, self.num_sc - 1] for _ in range(self.max_users)])

        power_action_space = spaces.Box(0.01, 1.0, shape=(self.max_users,), dtype=np.float32)
        offloading_action_space = MultiDiscrete([[0, self.ppo_config.split_quantization] for _ in range(self.n_sbs*self.max_users)])
        mec_power_action_space = spaces.Box(0.01, 1.0, shape=(self.max_users*self.n_sbs,), dtype=np.float32)
        self.action_spaces=[sc_action_space,offloading_action_space,mec_power_action_space,power_action_space]
        # ----- End  Action Specs----- #


    def get_available_action_shape(self,action_type):
        return self.action_mask.shape[1:]


    def get_observation_space(self,observation_type):
        obs_dim=self.get_observation().shape[1]
        return spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)

    def get_state_space(self,state_type):
        state_dim=self.get_global_state().shape[1]
        return spaces.Box(low=-np.inf, high=+np.inf, shape=(state_dim,), dtype=np.float32)

    def get_action_space(self,action_type):
        return self.action_spaces

    def reset(self, seed=None, options=None,step_type=0):
        super().reset(seed,options)
        info={}
        action_mask=self.action_mask
        return self.get_observation(),self.get_global_state(),action_mask,info


    def get_global_state(self):
        state=torch.empty((self.n_sbs,0),device=self.device)
        cur_second = int(self.cur_timeslot // self.slots_per_coherence)
        ready_to_transmit = self.bg_users.int()
        not_ready = ready_to_transmit == 0
        users_gains = self.normalized_gain[:, :, cur_second % self.channel_buffer_size].clone()
        cur_users_gains = 10 * torch.log10(users_gains)
        min_gain = torch.min(cur_users_gains)
        users_gains = (cur_users_gains - min_gain) / (torch.max(cur_users_gains) - min_gain)
        users_gains[not_ready] = -1
        state = torch.concat((state,users_gains), dim=1)
        state = torch.cat((state,self.ue_arrival_buffer, self.task_type_buffer), dim=1)
        user_rates = self.user_rates/ self.max_data_rate
        state = torch.cat((state, user_rates), dim=1)
        local_queue_delay = self.compute_local_delay()
        local_processing_delay = torch.ceil((self.ue_arrival_buffer * self.ue_config.task_power) / self.ue_computing_power)
        local_delay = local_queue_delay + local_processing_delay
        state = torch.cat((state, local_delay), dim=1)
        state = torch.cat((state, (self.ue_computing_power / self.max_ue_computing_power)), dim=1)
        state=state.reshape((1,-1))
        norm_mec_power=(self.mec_computing_power/self.max_mec_computing_power).reshape((1,-1))
        state= torch.cat((state,norm_mec_power),dim=1)
        return state.cpu().numpy()

    def get_observation(self):
        state = torch.empty((self.n_sbs, 0), device=self.device)
        cur_second = int(self.cur_timeslot // self.slots_per_coherence)
        ready_to_transmit = self.bg_users.int()
        not_ready = ready_to_transmit == 0
        users_gains = self.normalized_gain[:, :, cur_second % self.channel_buffer_size].clone()
        cur_users_gains = 10 * torch.log10(users_gains)
        min_gain = torch.min(cur_users_gains)
        users_gains = (cur_users_gains - min_gain) / (torch.max(cur_users_gains) - min_gain)
        users_gains[not_ready] = -1
        state = torch.concat((state, users_gains), dim=1)
        state = torch.cat((state, self.ue_arrival_buffer, self.task_type_buffer), dim=1)
        user_rates = self.user_rates / self.max_data_rate
        state = torch.cat((state, user_rates), dim=1)
        local_queue_delay = self.compute_local_delay()
        local_processing_delay = torch.ceil(
            (self.ue_arrival_buffer * self.ue_config.task_power) / self.ue_computing_power)
        local_delay = local_queue_delay + local_processing_delay
        state = torch.cat((state, local_delay), dim=1)
        state = torch.cat((state, (self.ue_computing_power / self.max_ue_computing_power)), dim=1)
        norm_mec_power = (self.mec_computing_power / self.max_mec_computing_power).transpose(0,1).expand(self.num_agents,-1)
        state = torch.cat((state, norm_mec_power), dim=1)
        return state.cpu().numpy()

    def step(self, action,step_type=0):
        noma_metrics=None
        if self.time_to_ra:
            sc_alloc = action[:,:self.max_users]
            assigned = self.assign_users_to_subchannels(torch.as_tensor(sc_alloc, dtype=torch.long, device=self.device))
            self.user_sc_assignment = assigned
            power_action = action[:, -self.max_users:]
            power_action = torch.clamp(power_action, 0.01, 1.0)
            power_action *= torch.as_tensor(self.action_mask[:, 0]).to(self.device)
            self.chosen_power_ratio = power_action.float()
            self.perform_noma_step()
            noma_metrics = self.metrics_tx
            self.prepare_next_noma_step()
        self.perform_early_steps()
        offloading_actions=(action[:, self.max_users:self.max_users+(self.max_users*self.n_sbs)]
                            .reshape((self.n_sbs,self.n_sbs,self.max_users))).transpose(1,2)
        split_actions = offloading_actions[:,:,0] / self.ppo_config.split_quantization
        self.task_split_ratio = torch.as_tensor(split_actions, dtype=self.task_split_ratio.dtype, device=self.device)
        idx = self.cur_timeslot % (self.max_task_deadline + 1)
        offloading_decision = offloading_actions[:,:,1:].reshape(self.n_sbs, self.max_users,
                                                    self.n_sbs - 1) / self.ppo_config.split_quantization
        self.task_offloading_decision[:, :, :-1, idx] = offloading_decision
        rem_split = 1 - split_actions-offloading_decision.sum(-1)
        rem_split *= self.ue_arrival_buffer > 0.0
        self.task_offloading_decision[:, :, -1, idx] = rem_split
        mec_power_ratio = action[:, self.max_users+(self.max_users*self.n_sbs):-self.max_users].reshape(self.n_sbs, self.max_users, self.n_sbs)
        self.task_mec_power_ratio[:, :, :, idx] = mec_power_ratio
        self.perform_later_steps()
        offloading_metrics = self.metrics_offloading
        info = {'noma_metrics':noma_metrics,'offloading_metrics': offloading_metrics}
        reward = self.offloading_reward.cpu().numpy()
        done = reward < -10
        self.time_to_ra = (self.cur_timeslot % self.slots_per_coherence) == 0 or self.instant_resource_offload
        if self.time_to_ra:
            self.action_mask[:, 0,:]=self.default_action_mask.copy()
        else:
            self.action_mask[:, 0]=np.zeros_like(self.action_mask[:, 0])
        self.action_mask[:,1] = (self.ue_arrival_buffer> 0).cpu().numpy()
        return self.get_observation(), self.get_global_state(), self.action_mask, reward, done, info




