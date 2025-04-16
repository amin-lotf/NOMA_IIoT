import numpy as np

from .distributions import Categorical, DiagGaussian, MultiVariate, NormalDist
import torch
import torch.nn as nn

from ...ppo_utils.util import check


class MixedACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, device, metadata: dict = None):
        super(MixedACTLayer, self).__init__()
        self.device = device
        action_outs=[]
        self.action_limits=[]
        tot_limit=0
        for act_space in action_space:
            act_space_type = act_space.__class__.__name__
            if act_space_type == 'Discrete' :
                action_dim = act_space.n
                tot_limit+=action_dim
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
        if metadata is not None:
            act_type = metadata.get('act_type')
            self.is_single_agent=metadata.get('is_single_agent')
            if act_type == 'noma':
                self.is_noma = True
                self.is_joint = False
                self.n_sc = metadata['n_sc']
                self.sc_capacity = metadata['sc_capacity']
                self.max_users = metadata['max_users']
            elif act_type =='joint':
                self.is_noma = False
                self.is_joint = True
                self.n_sc = metadata['n_sc']
                self.sc_capacity = metadata['sc_capacity']
                self.max_users = metadata['max_users']
                self.num_splits = metadata['split_quantization']

    def forward(self, x, available_actions=None, deterministic=None):
        actions = []
        mask_index=0
        disc_action_log_probs = []
        sc_stat = torch.zeros((x.shape[0], self.n_sc)).to(self.device)
        if self.is_joint:
            split_stat = torch.arange(self.num_splits + 1, device=self.device).reshape((1,1,-1)).expand(x.shape[0],self.max_users, -1)
            taken_splits = torch.zeros((x.shape[0],self.max_users, 1)).to(self.device)
        for idx, action_out in enumerate(self.action_outs):
            if idx < self.action_limits[0]:
                #------Subchannel section--------#
                act_mask = sc_stat < self.sc_capacity
                action_logit = action_out(x, act_mask)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                if available_actions is not None:
                    avail_act = available_actions[:,0, idx:idx + 1]
                    action = action * avail_act + (1 - avail_act) * -1
                    action_log_prob = action_log_prob * avail_act
                actions.append(action)
                disc_action_log_probs.append(action_log_prob)
                row_indices = torch.arange(sc_stat.shape[0]).to(self.device).reshape((-1,1))
                mask = action != -1
                sc_stat[row_indices[mask], action[mask]] += 1
            elif  self.is_joint and self.action_limits[0] <= idx < self.action_limits[1]:
                # ------Offloading decision section--------#
                act_idx = idx % self.max_users
                rem_splits = self.num_splits - taken_splits
                act_mask = split_stat <= rem_splits
                action_logit = action_out(x,act_mask[:,act_idx,:])
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                if available_actions is not None:
                    avail_act = available_actions[:, 1, act_idx:act_idx + 1]
                    action = action * avail_act
                    action_log_prob = action_log_prob * available_actions[:, 1, act_idx:act_idx + 1]
                actions.append(action)
                disc_action_log_probs.append(action_log_prob)
                taken_splits[:,act_idx,:] += action
            else:
                # ------Tx power section--------#
                if available_actions is not None:
                    action_logits = action_out(x, available_actions[:, 2,:])
                    cont_action = action_logits.mode() if deterministic else action_logits.sample()
                    cont_action = cont_action * available_actions[:, 2,:]
                else:
                    action_logits = action_out(x)
                    cont_action = action_logits.mode() if deterministic else action_logits.sample()
                actions.append(cont_action)
                cont_action_log_prob = action_logits.log_probs(cont_action)
        actions = torch.cat(actions, -1)
        disc_action_log_probs = torch.sum(torch.cat(disc_action_log_probs, dim=-1), dim=-1, keepdim=True)
        action_log_probs = disc_action_log_probs + cont_action_log_prob
        return actions, action_log_probs, available_actions

    #     return action_probs

    def evaluate_actions(self, x, action, available_actions: torch.Tensor = None):
        sc_action = torch.transpose(action[:, :self.max_users], 0, 1).int()
        disc_action_log_probs = []
        disc_entropy = []
        sc_stat = torch.zeros((x.shape[0], self.n_sc)).to(self.device)
        # available_actions = torch.randint(0, 2, (x.shape[0], self.n_sc * self.sc_capacity)).to(self.device)
        for idx, act_info in enumerate(zip(self.action_outs[:self.action_limits[0]], sc_action)):
            action_out, act = act_info
            act_mask = sc_stat < self.sc_capacity
            # act_mask = check(act_mask).to(self.device)
            action_logit = action_out(x, act_mask)
            act[act==-1]=0 # we change to 0 to avoid error, then use abailable_actions to zero out log_probs
            action_log_prob = action_logit.log_probs(act)
            entropy = action_logit.entropy().mean()
            if available_actions is not None:
                # action_log_prob = available_actions[:, idx:idx + 1]
                # entropy = action_logit.entropy()[available_actions[:, idx] == 1].mean()
                action_log_prob = action_log_prob*available_actions[:,0, idx:idx + 1]
                entropy = entropy*available_actions[:,0, idx:idx + 1]
            disc_entropy.append(entropy)
            disc_action_log_probs.append(action_log_prob)
            # act = act.cpu().numpy()
            row_indices = torch.arange(sc_stat.shape[0]).to(self.device)
            mask = act != -1
            sc_stat[row_indices[mask], act[mask]] += 1
        if self.is_joint:
            offloading_action = torch.transpose(action[:, self.action_limits[0]:self.action_limits[1]], 0, 1).int()
            split_stat = torch.arange(self.num_splits + 1, device=self.device).reshape((1, 1, -1)).expand(x.shape[0],
                                                                                                          self.max_users,
                                                                                                          -1)
            taken_splits = torch.zeros((x.shape[0], self.max_users, 1)).to(self.device)
            for idx, act_info in enumerate(zip(self.action_outs[self.action_limits[0]: self.action_limits[1]], offloading_action)):
                act_idx = idx % self.max_users
                action_out, act = act_info
                rem_splits = self.num_splits - taken_splits
                act_mask = split_stat <= rem_splits
                action_logit = action_out(x,act_mask[:,act_idx,:])
                action_log_prob = action_logit.log_probs(act)
                entropy = action_logit.entropy().mean()
                if available_actions is not None:
                    action_log_prob = action_log_prob*available_actions[:,1, act_idx:act_idx+1]
                    entropy = entropy*available_actions[:,1, act_idx:act_idx+1]
                disc_entropy.append(entropy)
                disc_action_log_probs.append(action_log_prob)
                taken_splits[:,act_idx,:] += act.unsqueeze(-1)
        disc_action_log_probs = torch.sum(torch.cat(disc_action_log_probs, dim=-1), dim=-1, keepdim=True)
        disc_entropy_t=torch.stack(disc_entropy)
        disc_entropy = torch.mean(disc_entropy_t[disc_entropy_t!=0])

        power_actions = action[:, self.action_limits[-2] :]
        # log_prob, entropy = self.action_out.evaluate_action(x,action)
        action_logits = self.action_outs[-1](x, available_actions[:,  2, :])
        cont_log_prob = action_logits.log_probs(power_actions)
        cont_entropy = action_logits.entropy().mean()
        log_prob = disc_action_log_probs + cont_log_prob
        entropy = disc_entropy + cont_entropy
        return log_prob, entropy
