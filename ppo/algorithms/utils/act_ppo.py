import numpy as np

from .distributions import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn


class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """

    def __init__(self, action_space, inputs_dim, use_orthogonal, gain,tpdv):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False
        self.tpdv = tpdv
        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            self.split_quant=action_space.high[0]+1
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continuous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList(
                [DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain), Categorical(
                    inputs_dim, discrete_dim, use_orthogonal, gain)])

    def forward(self, x, available_actions=None, deterministic=False):
        if self.mixed_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                actioDiagGaussiann_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)

        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            if available_actions is not None:
                cur_limit = 0
                split_stat = np.ones((x.shape[0], 1))
                for idx, action_out in enumerate(self.action_outs):
                    next_limit = action_out.linear.out_features
                    # action_logit = action_out(x,available_actions[:,idx,cur_limit:cur_limit+next_limit])
                    avail_acts=available_actions[:, cur_limit:cur_limit + next_limit]
                    action_logit = action_out(x, avail_acts)
                    available_actions[:, cur_limit:cur_limit + next_limit]=avail_acts
                    action = action_logit.mode() if deterministic else action_logit.sample()
                    action_log_prob = action_logit.log_probs(action)
                    actions.append(action)
                    action_log_probs.append(action_log_prob)
                    cur_limit += next_limit
                    chosen_ratio = np.round(action.cpu().numpy()/(self.split_quant-1),4)
                    # row_indices = np.arange(split_stat.shape[0])
                    split_stat=np.round(split_stat-chosen_ratio,4)
            else:
                split_stat = np.zeros((x.shape[0], 1))
                for action_out in self.action_outs:
                    # act_mask=
                    action_logit = action_out(x)
                    action = action_logit.mode() if deterministic else action_logit.sample()
                    action_log_prob = action_logit.log_probs(action)
                    actions.append(action)
                    action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)

        else:
            action_logits = self.action_out(x, available_actions)
            actions = action_logits.mode() if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)

        return actions, action_log_probs,available_actions

    def get_probs(self, x, available_actions=None):
        """
        Compute action probabilities from inputs.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)

        :return action_probs: (torch.Tensor)
        """
        if self.mixed_action or self.multi_discrete:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        else:
            action_logits = self.action_out(x, available_actions)
            action_probs = action_logits.probs

        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.mixed_action:
            a, b = action.split((2, 1), -1)
            b = b.long()
            action = [a, b]
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    if len(action_logit.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append((action_logit.entropy() * active_masks).sum() / active_masks.sum())
                    else:
                        dist_entropy.append(
                            (action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0] / 2.0 + dist_entropy[1] / 0.98  # ! dosen't make sense

        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            cur_limit = 0
            # split_stat = np.ones((x.shape[0], 1))
            for action_out, act in zip(self.action_outs, action):
                if available_actions is not None:
                    next_limit = action_out.linear.out_features
                    avail_acts = available_actions[:, cur_limit:cur_limit + next_limit]
                    # possible_ratios = np.round(
                    #     np.repeat(np.arange(self.split_quant).reshape(1, -1), avail_acts.shape[0], axis=0) / (
                    #                 self.split_quant - 1), 4)
                    # avail_ratios = possible_ratios <= split_stat
                    # act_mask = check(avail_ratios).to(**self.tpdv)
                    # avail_acts = avail_acts * act_mask
                    action_logit = action_out(x, avail_acts)
                    cur_limit += next_limit
                    # chosen_ratio = np.round(action.cpu().numpy() / (self.split_quant - 1), 4)
                    # row_indices = np.arange(split_stat.shape[0])
                    # split_stat = np.round(split_stat - chosen_ratio, 4)
                else:
                    action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())


            action_log_probs = torch.cat(action_log_probs, -1)  # ! could be wrong
            dist_entropy = sum(dist_entropy) / len(dist_entropy)

        else:
            action_logits = self.action_out(x, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy
