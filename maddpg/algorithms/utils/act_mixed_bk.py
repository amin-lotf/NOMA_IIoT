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
            # New flag for MADDPG mode and its noise parameters:

        else:
            self.is_single_agent = False
            self.maddpg = False

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

    def forward(self, x, available_actions=None, deterministic=None):
        actions = []
        disc_action_log_probs = []

        # ----- SINGLE AGENT MODE -----
        if self.is_single_agent:
            bs_count = self.n_sbs // self.num_agents  # number of base stations per tx agent
            total_decisions = self.max_users * bs_count  # total discrete decisions

            # --- Subchannel Allocation ---
            sc_stat = torch.zeros((x.shape[0], bs_count, self.n_sc), device=self.device)
            for idx in range(total_decisions):
                bs_idx = idx // self.max_users
                act_mask = sc_stat[:, bs_idx, :] < self.sc_capacity
                action_logit = self.action_outs[idx](x, act_mask)
                # Use mode/sample as usual
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                if self.maddpg:
                    action = self.add_maddpg_noise(action, action_logit, type='Categorical')
                if available_actions is not None:
                    avail_act = available_actions[:, 0, idx: idx + 1]
                    action = action * avail_act + (1 - avail_act) * -1
                    action_log_prob = action_log_prob * avail_act
                actions.append(action)
                disc_action_log_probs.append(action_log_prob)
                act = action.view(-1)
                mask = act != -1
                if mask.any():
                    valid_idx = torch.nonzero(mask, as_tuple=True)[0]
                    valid_act = act[mask].long()
                    sc_stat[valid_idx, bs_idx, valid_act] += 1

            # --- Task Offloading Decision (if joint mode) ---
            if self.is_joint:
                offload_total = total_decisions
                split_stat = torch.arange(self.num_splits + 1, device=self.device) \
                    .reshape((1, 1, -1)).expand(x.shape[0], bs_count, -1)
                taken_splits = torch.zeros((x.shape[0], bs_count, 1), device=self.device)
                for idx in range(offload_total):
                    bs_idx = idx // self.max_users
                    rem_splits = self.num_splits - taken_splits[:, bs_idx, :]
                    act_mask = split_stat[:, bs_idx, :] <= rem_splits
                    action_logit = self.action_outs[total_decisions + idx](x, act_mask)
                    action = action_logit.mode() if deterministic else action_logit.sample()
                    action_log_prob = action_logit.log_probs(action)
                    if self.maddpg:
                        action = self.add_maddpg_noise(action, action_logit,type='Categorical')
                    if available_actions is not None:
                        avail_act = available_actions[:, 1, idx: idx + 1]
                        action = action * avail_act
                        action_log_prob = action_log_prob * avail_act
                    actions.append(action)
                    disc_action_log_probs.append(action_log_prob)
                    taken_splits[:, bs_idx, :] += action

            # --- Tx Power (Continuous) ---
            cont_idx = -1
            if available_actions is not None:
                action_logits = self.action_outs[cont_idx](x, available_actions[:, 2, :])
                cont_action = action_logits.mode() if deterministic else action_logits.sample()
                if self.maddpg:
                    cont_action = self.add_maddpg_noise(cont_action, action_logits, type='Normal')
                cont_action = cont_action * available_actions[:, 2, :]
            else:
                action_logits = self.action_outs[cont_idx](x)
                cont_action = action_logits.mode() if deterministic else action_logits.sample()
                if self.maddpg:
                    cont_action = self.add_maddpg_noise(cont_action, action_logits, type='Normal')
            actions.append(cont_action)
            cont_action_log_prob = action_logits.log_probs(cont_action)

            disc_action_log_probs = torch.sum(torch.cat(disc_action_log_probs, dim=-1), dim=-1, keepdim=True)
            action_log_probs = disc_action_log_probs + cont_action_log_prob
            return torch.cat(actions, -1), action_log_probs, available_actions

        # ----- MULTI-AGENT MODE (Original) -----
        else:
            sc_stat = torch.zeros((x.shape[0], self.n_sc), device=self.device)
            if self.is_joint:
                split_stat = torch.arange(self.num_splits + 1, device=self.device) \
                    .reshape((1, 1, -1)).expand(x.shape[0], self.max_users, -1)
                taken_splits = torch.zeros((x.shape[0], self.max_users, 1), device=self.device)
            for idx, action_out in enumerate(self.action_outs):
                if idx < self.action_limits[0]:
                    # --- Subchannel Allocation ---
                    act_mask = sc_stat < self.sc_capacity
                    action_logit = action_out(x, act_mask)
                    action = action_logit.mode() if deterministic else action_logit.sample()
                    action_log_prob = action_logit.log_probs(action)
                    if self.maddpg:
                        action = self.add_maddpg_noise(action, action_logit,type='Categorical')
                    if available_actions is not None:
                        avail_act = available_actions[:, 0, idx:idx + 1]
                        action = action * avail_act + (1 - avail_act) * -1
                        action_log_prob = action_log_prob * avail_act
                    actions.append(action)
                    disc_action_log_probs.append(action_log_prob)
                    row_indices = torch.arange(sc_stat.shape[0], device=self.device).unsqueeze(-1)
                    mask = action != -1
                    sc_stat[row_indices[mask], action[mask]] += 1
                elif self.is_joint and self.action_limits[0] <= idx < self.action_limits[1]:
                    # --- Task Offloading Decision ---
                    act_idx = idx % self.max_users
                    rem_splits = self.num_splits - taken_splits
                    act_mask = split_stat <= rem_splits
                    action_logit = action_out(x, act_mask[:, act_idx, :])
                    action = action_logit.mode() if deterministic else action_logit.sample()
                    action_log_prob = action_logit.log_probs(action)
                    if self.maddpg:
                        action = self.add_maddpg_noise(action, action_logit,type='Categorical')
                    if available_actions is not None:
                        avail_act = available_actions[:, 1, act_idx:act_idx + 1]
                        action = action * avail_act
                        action_log_prob = action_log_prob * avail_act
                    actions.append(action)
                    disc_action_log_probs.append(action_log_prob)
                    taken_splits[:, act_idx, :] += action
                else:
                    # --- Tx Power (Continuous) ---
                    if available_actions is not None:
                        action_logits = action_out(x, available_actions[:, 2, :])
                        cont_action = action_logits.mode() if deterministic else action_logits.sample()
                        if self.maddpg:
                            cont_action = self.add_maddpg_noise(cont_action, action_logits, type='Normal')
                        cont_action = cont_action * available_actions[:, 2, :]
                    else:
                        action_logits = action_out(x)
                        cont_action = action_logits.mode() if deterministic else action_logits.sample()
                        if self.maddpg:
                            cont_action = self.add_maddpg_noise(cont_action, action_logits, type='Normal')
                    actions.append(cont_action)
                    cont_action_log_prob = action_logits.log_probs(cont_action)
            disc_action_log_probs = torch.sum(torch.cat(disc_action_log_probs, dim=-1), dim=-1, keepdim=True)
            action_log_probs = disc_action_log_probs + cont_action_log_prob
            return torch.cat(actions, -1), action_log_probs, available_actions

    def evaluate_actions(self, x, action, available_actions: torch.Tensor = None):
        # (Keep the evaluate_actions method as is  note that in MADDPG you typically
        # don't use the log probability for updating a deterministic actor.)
        if self.is_single_agent:
            bs_count = self.n_sbs // self.num_agents
            total_decisions = self.max_users * bs_count
            disc_action_log_probs = []
            disc_entropy = []
            sc_stat = torch.zeros((x.shape[0], bs_count, self.n_sc), device=self.device)
            subch_actions = torch.transpose(action[:, :total_decisions], 0, 1).int()
            for idx in range(total_decisions):
                bs_idx = idx // self.max_users
                act_mask = sc_stat[:, bs_idx, :] < self.sc_capacity
                action_logit = self.action_outs[idx](x, act_mask)
                act = subch_actions[idx]
                act[act == -1] = 0
                action_log_prob = action_logit.log_probs(act)
                entropy = action_logit.entropy().mean()
                if available_actions is not None:
                    action_log_prob = action_log_prob * available_actions[:, 0, idx:idx + 1]
                    entropy = entropy * available_actions[:, 0, idx:idx + 1]
                disc_entropy.append(entropy)
                disc_action_log_probs.append(action_log_prob)
                act = act.view(-1)
                mask = act != -1
                if mask.any():
                    valid_idx = torch.nonzero(mask, as_tuple=True)[0]
                    valid_act = act[mask].long()
                    sc_stat[valid_idx, bs_idx, valid_act] += 1

            if self.is_joint:
                offload_total = total_decisions
                offloading_actions = torch.transpose(action[:, total_decisions:total_decisions * 2], 0, 1).int()
                split_stat = torch.arange(self.num_splits + 1, device=self.device) \
                    .reshape((1, 1, -1)).expand(x.shape[0], bs_count, -1)
                taken_splits = torch.zeros((x.shape[0], bs_count, 1), device=self.device)
                for idx in range(offload_total):
                    bs_idx = idx // self.max_users
                    act = offloading_actions[idx]
                    rem_splits = self.num_splits - taken_splits[:, bs_idx, :]
                    act_mask = split_stat[:, bs_idx, :] <= rem_splits
                    action_logit = self.action_outs[total_decisions + idx](x, act_mask)
                    action_log_prob = action_logit.log_probs(act)
                    entropy = action_logit.entropy().mean()
                    if available_actions is not None:
                        action_log_prob = action_log_prob * available_actions[:, 1, idx:idx + 1]
                        entropy = entropy * available_actions[:, 1, idx:idx + 1]
                    disc_entropy.append(entropy)
                    disc_action_log_probs.append(action_log_prob)
                    taken_splits[:, bs_idx, :] += act.unsqueeze(-1)

            cont_idx = -1
            power_actions = action[:, self.action_limits[-2]:]
            if available_actions is not None:
                action_logits = self.action_outs[cont_idx](x, available_actions[:, 2, :])
                cont_log_prob = action_logits.log_probs(power_actions)
            else:
                action_logits = self.action_outs[cont_idx](x)
                cont_log_prob = action_logits.log_probs(power_actions)
            disc_action_log_probs = torch.sum(torch.cat(disc_action_log_probs, dim=-1), dim=-1, keepdim=True)
            disc_entropy = torch.mean(torch.stack(disc_entropy))
            log_prob = disc_action_log_probs + cont_log_prob
            entropy = disc_entropy + action_logits.entropy().mean()
            return log_prob, entropy

        else:
            sc_action = torch.transpose(action[:, :self.max_users], 0, 1).int()
            disc_action_log_probs = []
            disc_entropy = []
            sc_stat = torch.zeros((x.shape[0], self.n_sc), device=self.device)
            for idx, (action_out, act) in enumerate(zip(self.action_outs[:self.action_limits[0]], sc_action)):
                act_mask = sc_stat < self.sc_capacity
                action_logit = action_out(x, act_mask)
                act[act == -1] = 0
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
            if self.is_joint:
                offloading_action = torch.transpose(action[:, self.action_limits[0]:self.action_limits[1]], 0, 1).int()
                split_stat = torch.arange(self.num_splits + 1, device=self.device) \
                    .reshape((1, 1, -1)).expand(x.shape[0], self.max_users, -1)
                taken_splits = torch.zeros((x.shape[0], self.max_users, 1), device=self.device)
                for idx, (action_out, act) in enumerate(
                        zip(self.action_outs[self.action_limits[0]: self.action_limits[1]], offloading_action)):
                    act_idx = idx % self.max_users
                    action_logit = action_out(x, split_stat[:, act_idx, :] <= (
                            self.num_splits - taken_splits[:, act_idx, :]))
                    action_log_prob = action_logit.log_probs(act)
                    entropy = action_logit.entropy().mean()
                    if available_actions is not None:
                        action_log_prob = action_log_prob * available_actions[:, 1, act_idx:act_idx + 1]
                        entropy = entropy * available_actions[:, 1, act_idx:act_idx + 1]
                    disc_entropy.append(entropy)
                    disc_action_log_probs.append(action_log_prob)
                    taken_splits[:, act_idx, :] += act.unsqueeze(-1)
            disc_action_log_probs = torch.sum(torch.cat(disc_action_log_probs, dim=-1), dim=-1, keepdim=True)
            disc_entropy_t = torch.stack(disc_entropy)
            disc_entropy = torch.mean(disc_entropy_t[disc_entropy_t != 0])
            power_actions = action[:, self.action_limits[-2]:]
            if available_actions is not None:
                action_logits = self.action_outs[-1](x, available_actions[:, 2, :])
                cont_log_prob = action_logits.log_probs(power_actions)
            else:
                action_logits = self.action_outs[-1](x)
                cont_log_prob = action_logits.log_probs(power_actions)
            cont_entropy = action_logits.entropy().mean()
            log_prob = disc_action_log_probs + cont_log_prob
            entropy = disc_entropy + cont_entropy
            return log_prob, entropy
