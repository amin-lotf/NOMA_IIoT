import numpy as np

from .distributions import Categorical, DiagGaussian, MultiVariate, NormalDist
import torch
import torch.nn as nn

from ...ppo_utils.util import check


class MixedACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, device, metadata: dict = None):
        super(MixedACTLayer, self).__init__()
        self.device = device
        disc_action_dims = action_space[0].high - action_space[0].low + 1
        disc_action_outs = []
        for action_dim in disc_action_dims:
            disc_action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))

        cont_action_dim = action_space[1].shape[0]
        # self.action_out = MultiVariate(inputs_dim, action_dim, action_space)
        # self.action_out = NormalDist(inputs_dim, action_dim, action_space)
        cont_action_out = DiagGaussian(inputs_dim, cont_action_dim, action_space[1])
        disc_action_outs.append(cont_action_out)
        self.action_outs = nn.ModuleList(disc_action_outs)
        if metadata is not None:
            self.n_sc = metadata['n_sc']
            self.sc_capacity = metadata['sc_capacity']
            self.n_sc_actions = len(disc_action_dims)

    def forward(self, x, available_actions=None, deterministic=None):
        actions = []
        disc_action_log_probs = []
        sc_stat = np.zeros((1, self.n_sc))
        for idx, action_out in enumerate(self.action_outs):
            if idx < self.n_sc_actions:
                act_mask = sc_stat < self.sc_capacity
                act_mask = check(act_mask).to(self.device)
                action_logit = action_out(x, act_mask)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                if available_actions is not None:
                    avail_act = available_actions[:, idx:idx + 1]
                    action = action * avail_act + (1 - avail_act) * -1
                    action_log_prob = action_log_prob * available_actions[:, idx:idx + 1]
                actions.append(action)
                disc_action_log_probs.append(action_log_prob)
                act = action.cpu().numpy()
                # row_indices = np.arange(sc_stat.shape[0])
                if act[:, 0] != -1:
                    sc_stat[0, act[:, 0]] += 1
            else:
                action_logits = action_out(x, available_actions)
                cont_action = action_logits.mode() if deterministic else action_logits.sample()
                if available_actions is not None:
                    cont_action = cont_action * available_actions
                actions.append(cont_action)
                cont_action_log_prob = action_logits.log_probs(cont_action)
        actions = torch.cat(actions, -1)
        disc_action_log_probs = torch.sum(torch.cat(disc_action_log_probs, dim=-1), dim=-1, keepdim=True)
        action_log_probs = disc_action_log_probs + cont_action_log_prob
        return actions, action_log_probs, available_actions

    #     return action_probs

    def evaluate_actions(self, x, action, available_actions: torch.Tensor = None):
        sc_action = torch.transpose(action[:, :self.n_sc_actions], 0, 1).int()
        disc_action_log_probs = []
        disc_entropy = []
        cur_limit = 0
        sc_stat = np.zeros((x.shape[0], self.n_sc))
        # available_actions = torch.randint(0, 2, (x.shape[0], self.n_sc * self.sc_capacity)).to(self.device)
        for idx, act_info in enumerate(zip(self.action_outs[:self.n_sc_actions], sc_action)):
            action_out, act = act_info
            act_mask = sc_stat < self.sc_capacity
            act_mask = check(act_mask).to(self.device)
            action_logit = action_out(x, act_mask)
            if available_actions is not None and available_actions[0, idx] == 0:
                action_log_prob = available_actions[:, idx:idx + 1]
                # entropy = action_logit.entropy()[available_actions[:, idx] == 1].mean()
            else:
                action_log_prob = action_logit.log_probs(act)
                entropy = action_logit.entropy().mean()
                disc_entropy.append(entropy)
            disc_action_log_probs.append(action_log_prob)
            act = act.cpu().numpy()
            row_indices = np.arange(sc_stat.shape[0])
            mask = act != -1
            sc_stat[row_indices[mask], act[mask]] += 1
        disc_action_log_probs = torch.sum(torch.cat(disc_action_log_probs, dim=-1), dim=-1, keepdim=True)
        disc_entropy = torch.mean(torch.stack(disc_entropy))

        power_actions = action[:, self.n_sc_actions:]
        # log_prob, entropy = self.action_out.evaluate_action(x,action)
        action_logits = self.action_outs[-1](x, available_actions)
        cont_log_prob = action_logits.log_probs(power_actions)
        cont_entropy = action_logits.entropy().mean()
        log_prob = disc_action_log_probs + cont_log_prob
        entropy = disc_entropy + cont_entropy
        return log_prob, entropy
