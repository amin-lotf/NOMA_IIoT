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
        if metadata is not None:
            act_type=metadata.get('act_type',False)
            self.is_single_agent = metadata.get('is_single_agent', False)
            self.num_agents = metadata.get('num_agents', False)
            self.n_sbs = metadata.get('n_sbs', False)
            if act_type== 'noma':
                self.is_noma=True
                self.is_offloading=False
                self.n_sc=metadata['n_sc']
                self.sc_capacity=metadata['sc_capacity']
                self.max_users=metadata['max_users']
            elif act_type=='offloading':
                self.is_noma = False
                self.is_offloading = True
                self.num_splits=metadata['split_quantization']



    def forward(self, x, available_actions=None, deterministic=False):
        actions = []
        action_log_probs = []
        if self.is_noma:
            sc_stat = torch.zeros((x.shape[0], self.n_sc)).to(self.device)
            for idx,action_out in enumerate(self.action_outs):
                if  idx < self.max_users:
                    act_mask = sc_stat < self.sc_capacity
                else:
                    act_mask = None
                action_logit = action_out(x,act_mask)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                if available_actions is not None:
                    avail_act = available_actions[:, idx:idx + 1]
                    action = action * avail_act + (1 - avail_act) * -1 if self.is_noma else action * avail_act
                    action_log_prob = action_log_prob * available_actions[:, idx:idx + 1]
                actions.append(action)
                action_log_probs.append(action_log_prob)
                if  idx < self.max_users:
                    row_indices = torch.arange(sc_stat.shape[0]).to(self.device).reshape((-1, 1))
                    mask = action != -1
                    sc_stat[row_indices[mask], action[mask]] += 1
        elif self.is_offloading:
            split_stat = torch.arange(self.num_splits+1,device=self.device).unsqueeze(0).expand(x.shape[0],-1)
            taken_splits = torch.zeros((x.shape[0], 1)).to(self.device)
            for idx,action_out in enumerate(self.action_outs):
                rem_splits=self.num_splits-taken_splits
                act_mask = split_stat <= rem_splits
                action_logit = action_out(x,act_mask)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                if available_actions is not None:
                    avail_act = available_actions[:, idx:idx + 1]
                    action = action * avail_act + (1 - avail_act) * -1 if self.is_noma else action * avail_act
                    action_log_prob = action_log_prob * available_actions[:, idx:idx + 1]
                actions.append(action)
                action_log_probs.append(action_log_prob)
                taken_splits+=action
        else:
            for idx,action_out in enumerate(self.action_outs):
                act_mask = None
                action_logit = action_out(x,act_mask)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
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
            sc_stat = torch.zeros((x.shape[0], self.n_sc)).to(self.device)
            # available_actions = torch.randint(0, 2, (x.shape[0], self.n_sc * self.sc_capacity)).to(self.device)
            for idx, act_info in enumerate(zip(self.action_outs[:self.max_users], sc_action)):
                action_out, act = act_info
                act_mask = sc_stat < self.sc_capacity
                # act_mask = check(act_mask).to(self.device)
                action_logit = action_out(x, act_mask)
                act[act == -1] = 0  # we change to 0 to avoid error, then use abailable_actions to zero out log_probs
                action_log_prob = action_logit.log_probs(act)
                entropy = action_logit.entropy().mean()
                if available_actions is not None:
                    # action_log_prob = available_actions[:, idx:idx + 1]
                    # entropy = action_logit.entropy()[available_actions[:, idx] == 1].mean()
                    action_log_prob = action_log_prob * available_actions[:, 0, idx:idx + 1]
                    entropy = entropy * available_actions[:, 0, idx:idx + 1]
                dist_entropy.append(entropy)
                action_log_probs.append(action_log_prob)
                # act = act.cpu().numpy()
                row_indices = torch.arange(sc_stat.shape[0]).to(self.device)
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
                    action_log_prob = action_log_prob * available_actions[:, 0, idx:idx + 1]
                    entropy = entropy * available_actions[:, 0, idx:idx + 1]
                dist_entropy.append(entropy)
                action_log_probs.append(action_log_prob)
        elif self.is_offloading:
            offloading_action = torch.transpose(action, 0, 1).int()
            split_stat = torch.arange(self.num_splits + 1, device=self.device).unsqueeze(0).expand(x.shape[0], -1)
            taken_splits = torch.zeros((x.shape[0], 1)).to(self.device)
            for idx, act_info in enumerate(zip(self.action_outs, offloading_action)):
                action_out, act = act_info
                rem_splits = self.num_splits - taken_splits
                act_mask = split_stat <= rem_splits
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
                taken_splits+=act.unsqueeze(-1)
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
