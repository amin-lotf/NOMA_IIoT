from .distributions import  Categorical
import torch
import torch.nn as nn



class MultiDiscreteACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, device,metadata:dict=None):
        super(MultiDiscreteACTLayer, self).__init__()
        self.device = device
        action_dims = action_space.high - action_space.low + 1
        self.action_outs = []
        for action_dim in action_dims:
            self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
        self.action_outs = nn.ModuleList(self.action_outs)
        self.maddpg = True
        self.noise_scale = metadata.get('noise_scale', 0.1)  # for continuous actions
        self.noise_epsilon = metadata.get('noise_epsilon', 0.1)  # for discrete actions
        if metadata is not None:
            act_type=metadata.get('act_type',False)
            self.is_single_agent = metadata.get('is_single_agent', False)
            self.num_agents = metadata.get('num_agents', False)
            self.n_sbs = metadata.get('n_sbs', False)
            self.max_users = metadata['max_users']
            if act_type== 'noma':
                self.is_noma=True
                self.is_offloading=False
                self.n_sc=metadata['n_sc']
                self.sc_capacity=metadata['sc_capacity']
            elif act_type=='offloading':
                self.is_noma = False
                self.is_offloading = True
                self.num_splits=metadata['split_quantization']




    def add_maddpg_noise(self, action,action_logit, type):
        # For discrete actions (FixedCategorical): use �greedy exploration.
        # For continuous actions (FixedNormal): add Gaussian noise.
        if type=='Categorical':  # checking underlying dist type
            # In practice, action_logit is created by our Categorical module and returns FixedCategorical.
            # Here we assume that if it is categorical then we apply �greedy:
            random_mask = (torch.rand_like(action.float()) < self.noise_epsilon).float()
            num_actions = action_logit.logits.size(-1)
            random_action = torch.randint(0, num_actions, action.shape, device=self.device)
            return random_mask * random_action + (1 - random_mask) * action
        elif type=='Normal':
            # For continuous actions add additive Gaussian noise.
            return action + torch.randn_like(action) * self.noise_scale
        else:
            return action



    def forward(self, x, available_actions=None, deterministic=False):
        actions = []
        action_log_probs = []

        if self.is_noma:
            sc_stat = torch.zeros((x.shape[0], self.n_sc)).to(self.device)
            for idx,action_out in enumerate(self.action_outs):
                if idx < self.max_users:
                    # --- Subchannel Allocation ---
                    act_mask = sc_stat < self.sc_capacity
                    action_logit = action_out(x, act_mask)
                    action = action_logit.mode() if deterministic else action_logit.sample()
                    action_log_prob = action_logit.log_probs(action)
                    if self.maddpg:
                        action = self.add_maddpg_noise(action, action_logit, type='Categorical')
                    if available_actions is not None:
                        avail_act = available_actions[:, 0, idx:idx + 1]
                        action = action * avail_act + (1 - avail_act) * -1
                        action_log_prob = action_log_prob * avail_act
                    actions.append(action)
                    action_log_probs.append(action_log_prob)
                    row_indices = torch.arange(sc_stat.shape[0], device=self.device).unsqueeze(-1)
                    mask = action != -1
                    sc_stat[row_indices[mask], action[mask]] += 1
                else:
                    act_idx = idx % self.max_users
                    action_logit = action_out(x)
                    action = action_logit.mode() if deterministic else action_logit.sample()
                    action_log_prob = action_logit.log_probs(action)
                    if self.maddpg:
                        action = self.add_maddpg_noise(action, action_logit, type='Categorical')
                    if available_actions is not None:
                        avail_act = available_actions[:, 2, act_idx:act_idx + 1]
                        action = action * avail_act + (1 - avail_act) * -1
                        action_log_prob = action_log_prob * avail_act
                    actions.append(action)
                    action_log_probs.append(action_log_prob)

        elif self.is_offloading:
            ue_count = self.n_sbs*self.max_users // self.num_agents  # number of ue controlled by an agent
            split_stat = torch.arange(self.num_splits + 1, device=self.device) \
                .reshape((1, 1, -1)).expand(x.shape[0], ue_count, -1)
            taken_splits = torch.zeros((x.shape[0], ue_count, 1), device=self.device)
            for idx,action_out in enumerate(self.action_outs):
                # In task offloading, n_sbs represents the task offloading decision( local split + number of SBSs-1)
                ue_idx = idx // self.n_sbs
                rem_splits = self.num_splits - taken_splits[:, ue_idx, :]
                act_mask = split_stat[:, ue_idx, :] <= rem_splits
                action_logit = action_out(x,act_mask)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                if self.maddpg:
                    action = self.add_maddpg_noise(action, action_logit, type='Categorical')
                if available_actions is not None:
                    avail_act = available_actions[:, idx:idx + 1]
                    action = action * avail_act + (1 - avail_act) * -1 if self.is_noma else action * avail_act
                    action_log_prob = action_log_prob * available_actions[:, idx:idx + 1]
                actions.append(action)
                action_log_probs.append(action_log_prob)
                taken_splits[:, ue_idx, :] += action
        else:
            for idx,action_out in enumerate(self.action_outs):
                act_mask = None
                action_logit = action_out(x,act_mask)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                if self.maddpg:
                    action = self.add_maddpg_noise(action, action_logit, type='Categorical')
                if available_actions is not None:
                    avail_act = available_actions[:, idx:idx + 1]
                    action = action * avail_act + (1 - avail_act) * -1 if self.is_noma else action * avail_act
                    action_log_prob = action_log_prob * available_actions[:, idx:idx + 1]
                actions.append(action)
                action_log_probs.append(action_log_prob)

        actions = torch.cat(actions, -1)
        action_log_probs = torch.sum(torch.cat(action_log_probs, dim=-1), dim=-1, keepdim=True)
        return actions, action_log_probs, available_actions


    def evaluate_actions(self, x, action, available_actions=None):
        action_log_probs = []
        dist_entropy = []
        if self.is_noma:
            sc_action = torch.transpose(action[:, :self.max_users], 0, 1).int()
            disc_action_log_probs = []
            disc_entropy = []
            sc_stat = torch.zeros((x.shape[0], self.n_sc), device=self.device)
            for idx, (action_out, act) in enumerate(zip(self.action_outs[:self.max_users], sc_action)):
                act_mask = sc_stat < self.sc_capacity
                action_logit = action_out(x, act_mask)
                act[act == -1] = 0  # avoid error in log_probs
                action_log_prob = action_logit.log_probs(act)
                entropy = action_logit.entropy().mean()
                if available_actions is not None:
                    action_log_prob = action_log_prob * available_actions[:, 0, idx:idx + 1]
                    entropy = entropy * available_actions[:, 0, idx:idx + 1]
                disc_entropy.append(entropy)
                disc_action_log_probs.append(action_log_prob)
                row_indices = torch.arange(sc_stat.shape[0], device=self.device).unsqueeze(-1)
                mask = act != -1
                sc_stat[row_indices[mask], act[mask]] += 1
            power_action = torch.transpose(action[:, self.max_users:], 0, 1).int()
            for idx, act_info in enumerate(zip(self.action_outs[self.max_users:], power_action)):
                action_out, act = act_info
                action_logit = action_out(x)
                act[act == -1] = 0  # we change to 0 to avoid error, then use abailable_actions to zero out log_probs
                action_log_prob = action_logit.log_probs(act)
                entropy = action_logit.entropy().mean()
                if available_actions is not None:
                    # action_log_prob = available_actions[:, idx:idx + 1]
                    # entropy = action_logit.entropy()[available_actions[:, idx] == 1].mean()
                    action_log_prob = action_log_prob * available_actions[:, 2, idx:idx + 1]
                    entropy = entropy * available_actions[:, 2, idx:idx + 1]
                dist_entropy.append(entropy)
                action_log_probs.append(action_log_prob)
        elif self.is_offloading:
            offloading_action = torch.transpose(action, 0, 1).int()
            ue_count = self.n_sbs * self.max_users // self.num_agents  # number of ue controlled by an agent
            split_stat = torch.arange(self.num_splits + 1, device=self.device) \
                .reshape((1, 1, -1)).expand(x.shape[0], ue_count, -1)
            taken_splits = torch.zeros((x.shape[0], ue_count, 1), device=self.device)
            for idx, act_info in enumerate(zip(self.action_outs, offloading_action)):
                action_out, act = act_info
                ue_idx = idx // self.n_sbs
                rem_splits = self.num_splits - taken_splits[:, ue_idx, :]
                act_mask = split_stat[:, ue_idx, :] <= rem_splits
                action_logit = action_out(x, act_mask)
                act[act == -1] = 0  # we change to 0 to avoid error, then use abailable_actions to zero out log_probs
                action_log_prob = action_logit.log_probs(act)
                entropy = action_logit.entropy().mean()
                if available_actions is not None:
                    # action_log_prob = available_actions[:, idx:idx + 1]
                    # entropy = action_logit.entropy()[available_actions[:, idx] == 1].mean()
                    action_log_prob = action_log_prob * available_actions[:,  idx:idx + 1]
                    entropy = entropy * available_actions[:,  idx:idx + 1]
                dist_entropy.append(entropy)
                action_log_probs.append(action_log_prob)
                # act = act.cpu().numpy()
                taken_splits[:, ue_idx, :] += act.unsqueeze(-1)
        else:
            action = torch.transpose(action, 0, 1)
            for idx, act_info in enumerate(zip(self.action_outs, action)):
                action_out, act = act_info
                act_mask = None
                action_logit = action_out(x, act_mask)
                if available_actions is not None and available_actions[0, idx] == 0:
                    action_log_prob = available_actions[:, idx:idx + 1]
                    # entropy = action_logit.entropy()[available_actions[:, idx] == 1].mean()
                else:
                    action_log_prob = action_logit.log_probs(act)
                    entropy = action_logit.entropy().mean()
                    dist_entropy.append(entropy)
                action_log_probs.append(action_log_prob)
        action_log_probs = torch.sum(torch.cat(action_log_probs, dim=-1), dim=-1, keepdim=True)
        dist_entropy = torch.mean(torch.stack(dist_entropy))
        return action_log_probs, dist_entropy
