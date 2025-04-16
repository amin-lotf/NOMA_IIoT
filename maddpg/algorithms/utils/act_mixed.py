import numpy as np
import torch
import torch.nn as nn
from .distributions import Categorical, DiagGaussian, MultiVariate, NormalDist



class MixedACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, device, metadata: dict = None):
        super(MixedACTLayer, self).__init__()
        self.device = device

        action_outs = []
        self.action_limits = []
        tot_limit = 0
        for act_space in action_space:
            act_space_type = act_space.__class__.__name__
            if act_space_type == 'Discrete':
                action_dim = act_space.n
                tot_limit += action_dim
                self.action_limits.append(tot_limit)
                action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            elif act_space_type == 'MultiDiscrete':
                multi_action_dims = act_space.high - act_space.low + 1
                for action_dim in multi_action_dims:
                    action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
                tot_limit += len(multi_action_dims)
                self.action_limits.append(tot_limit)
            elif act_space_type == 'Box':
                action_dim = act_space.shape[0]
                tot_limit += action_dim
                self.action_limits.append(tot_limit)
                action_outs.append(DiagGaussian(inputs_dim, action_dim, act_space))

        self.action_outs = nn.ModuleList(action_outs)

        self.maddpg = True
        self.noise_scale = metadata.get('noise_scale', 0.1)  # for continuous actions
        self.noise_epsilon = metadata.get('noise_epsilon', 0.1)  # for discrete actions
        # Process metadata for extra functionality (including MADDPG)
        if metadata is not None:
            act_type = metadata.get('act_type', False)
            self.is_single_agent = metadata.get('is_single_agent', False)
            self.num_agents = metadata.get('num_agents', False)
            self.n_sbs = metadata.get('n_sbs', False)
            if act_type == 'noma':
                self.is_noma = True
                self.is_joint = False
                self.n_sc = metadata['n_sc']
                self.sc_capacity = metadata['sc_capacity']
                self.max_users = metadata['max_users']
            elif act_type == 'joint':
                self.is_noma = False
                self.is_joint = True
                self.n_sc = metadata['n_sc']
                self.sc_capacity = metadata['sc_capacity']
                self.max_users = metadata['max_users']
                self.num_splits = metadata['split_quantization']
            elif act_type == 'offloading':
                self.is_noma = False
                self.is_joint = False
                self.is_offloading = True
                self.num_agents = metadata.get('num_agents', False)
                self.n_sbs = metadata.get('n_sbs', False)
                self.max_users = metadata['max_users']
                self.num_splits = metadata['split_quantization']
            # New flag for MADDPG mode and its noise parameters:

        else:
            self.is_single_agent = False
            self.maddpg = False

    def add_maddpg_noise(self, action, action_logit, type, act_mask=None):
        # For discrete actions (FixedCategorical): use ε-greedy exploration.
        # For continuous actions (FixedNormal): add Gaussian noise.
        if type == 'Categorical':
            random_mask = (torch.rand_like(action.float()) < self.noise_epsilon).int()
            if act_mask is not None:
                # Assume act_mask is a bool tensor of shape (batch_size, num_actions)
                # Convert mask to float so that allowed actions have nonzero probability.
                allowed_probs = act_mask.float()
                # Sample one valid action per batch element.
                random_action = torch.multinomial(allowed_probs, num_samples=1)
                # Squeeze extra dimension if needed to match the shape of action.
                if random_action.dim() > action.dim():
                    random_action = random_action.squeeze(-1)
            else:
                num_actions = action_logit.logits.size(-1)
                random_action = torch.randint(0, num_actions, action.shape, device=self.device)
            # Replace action with a randomly chosen valid one where random_mask is 1.
            return random_mask * random_action + (1 - random_mask) * action

        elif type == 'Normal':
            # For continuous actions add additive Gaussian noise.
            return torch.clamp(action + torch.randn_like(action) * self.noise_scale,0.01,1.0)
        else:
            return action

    def forward(self, x, available_actions=None,noise_scale=None,noise_epsilon=None, deterministic=None):
        if noise_scale:
            self.noise_scale=noise_scale
        if noise_epsilon:
            self.noise_epsilon=noise_epsilon
        actions = []
        disc_action_log_probs = []
        if self.is_noma:
            sc_stat = torch.zeros((x.shape[0], self.n_sc), device=self.device)
            for idx, action_out in enumerate(self.action_outs):
                if idx < self.action_limits[0]:
                    # --- Subchannel Allocation ---
                    act_mask = sc_stat < self.sc_capacity
                    action_logit = action_out(x, act_mask)
                    action =  action_logit.sample()
                    action_log_prob = action_logit.log_probs(action)
                    if self.maddpg and not deterministic:
                        action = self.add_maddpg_noise(action, action_logit, type='Categorical',act_mask=act_mask)
                    if available_actions is not None:
                        avail_act = available_actions[:,idx:idx + 1]
                        action = action * avail_act + (1 - avail_act) * -1
                        action_log_prob = action_log_prob * avail_act
                    actions.append(action)
                    disc_action_log_probs.append(action_log_prob)
                    row_indices = torch.arange(sc_stat.shape[0], device=self.device).unsqueeze(-1)
                    mask = action != -1
                    sc_stat[row_indices[mask], action[mask]] += 1
                else:
                    # --- Tx Power (Continuous) ---
                    if available_actions is not None:
                        action_logits = action_out(x, available_actions[:,  :])
                        cont_action =  action_logits.sample()
                        if self.maddpg and not deterministic:
                            cont_action = self.add_maddpg_noise(cont_action, action_logits, type='Normal')
                        cont_action = cont_action * available_actions[:,  :]
                    else:
                        action_logits = action_out(x)
                        cont_action =  action_logits.sample()
                        if self.maddpg and not deterministic:
                            cont_action = self.add_maddpg_noise(cont_action, action_logits, type='Normal')
                    actions.append(cont_action)
                    cont_action_log_prob = action_logits.log_probs(cont_action)
            disc_action_log_probs = torch.sum(torch.cat(disc_action_log_probs, dim=-1), dim=-1, keepdim=True)
            action_log_probs = disc_action_log_probs + cont_action_log_prob
            return torch.cat(actions, -1), action_log_probs, available_actions
        else:
            ue_count = self.n_sbs * self.max_users // self.num_agents  # number of ue controlled by an agent
            split_stat = torch.arange(self.num_splits + 1, device=self.device) \
                .reshape((1, 1, -1)).expand(x.shape[0], ue_count, -1)
            taken_splits = torch.zeros((x.shape[0], ue_count, 1), device=self.device)
            for idx, action_out in enumerate(self.action_outs):
                if idx < self.action_limits[0]:
                    # In task offloading, n_sbs represents the task offloading decision( local split + number of SBSs-1)
                    ue_idx = idx // self.n_sbs
                    rem_splits = self.num_splits - taken_splits[:, ue_idx, :]
                    act_mask = split_stat[:, ue_idx, :] <= rem_splits
                    action_logit = action_out(x, act_mask)
                    action =  action_logit.sample()
                    action_log_prob = action_logit.log_probs(action)
                    if self.maddpg and not deterministic:
                        action = self.add_maddpg_noise(action, action_logit, type='Categorical',act_mask=act_mask)
                    if available_actions is not None:
                        avail_act = available_actions[:, idx:idx + 1]
                        action = action * avail_act + (1 - avail_act) * -1 if self.is_noma else action * avail_act
                        action_log_prob = action_log_prob * available_actions[:, idx:idx + 1]
                    actions.append(action)
                    disc_action_log_probs.append(action_log_prob)
                    taken_splits[:, ue_idx, :] += action
                else:
                    # --- MEC Power (Continuous) ---
                    if available_actions is not None:
                        action_logits = action_out(x, available_actions)
                        cont_action =  action_logits.sample()
                        if self.maddpg and not deterministic:
                            cont_action = self.add_maddpg_noise(cont_action, action_logits, type='Normal')
                        cont_action = cont_action * available_actions
                    else:
                        action_logits = action_out(x)
                        cont_action = action_logits.sample()
                        if self.maddpg and not deterministic:
                            cont_action = self.add_maddpg_noise(cont_action, action_logits, type='Normal')
                    actions.append(cont_action)
                    cont_action_log_prob = action_logits.log_probs(cont_action)
            disc_action_log_probs = torch.sum(torch.cat(disc_action_log_probs, dim=-1), dim=-1, keepdim=True)
            action_log_probs = disc_action_log_probs + cont_action_log_prob
            return torch.cat(actions, -1), action_log_probs, available_actions

