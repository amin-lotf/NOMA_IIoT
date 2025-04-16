import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from isac.sac_algorithms.utlis.util import init

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""

#
# Standardize distribution interfaces
#
class LinearDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(LinearDQN, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return x


class LinearDuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(LinearDuelingDQN, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc = nn.Sequential(
            Linear(num_inputs, num_inputs),
            nn.ReLU(),
            Linear(num_inputs, num_outputs)
        )

        self.fc_adv = Linear(num_inputs, 1)



    def forward(self, x, available_actions=None):
        adv = self.fc_adv(x)
        val = self.fc(x)
        res = val + (adv - adv.mean(dim=1, keepdim=True))
        if available_actions is not None:
            res[available_actions == 0] = -1e10
        return res


# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def __init__(self,mean,log_std,action_scale,action_bias, mask=None):
        super(FixedNormal,self).__init__(mean,log_std)
        self.mask = mask  # shape: [batch_size, num_outputs] or [num_outputs]
        self.action_scale=action_scale
        self.action_bias=action_bias
    def log_probs(self, actions):
        eps = 1e-6
        x = torch.clamp((actions - self.action_bias) / self.action_scale, -1 + eps, 1 - eps)
        # raw_action = 0.5 * torch.log((1 + x) / (1 - x))  # or torch.atanh(x)
        raw_action = torch.atanh(x)
        # raw_action = torch.atanh((x - self.action_bias) / self.action_scale)
        log_prob = super().log_prob(raw_action).sum(dim=-1,keepdim=True)
        log_prob -= torch.sum(torch.log(self.action_scale * (1 - torch.tanh(raw_action).pow(2)) + 1e-6), dim=-1,keepdim=True)
        if self.mask is not None:
            # multiply by the mask so invalid dims get 0 contribution
            log_prob = log_prob * self.mask
        return log_prob

    def entropy(self):
        # Same idea for entropy
        ents = super().entropy()
        if self.mask is not None:
            ents = ents * self.mask
        return ents.sum(-1)

    def mode(self):
        return self.mean

    def sample(self, sample_shape=torch.Size()):
        raw_action = super().rsample()  # Reparameterization trick
        action = torch.tanh(raw_action) * self.action_scale + self.action_bias
        return action


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, action_space,use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))
        if action_space is None:
            self.action_scale = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
            self.action_bias = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)
        else:
            self.action_scale = torch.nn.Parameter(torch.FloatTensor(
                (action_space.high - action_space.low) / 2.), requires_grad=False)
            self.action_bias = torch.nn.Parameter(torch.FloatTensor(
                (action_space.high + action_space.low) / 2.), requires_grad=False)

    def forward(self, x,available_actions=None):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        if available_actions is not None:
            # Force mean = 0 for invalid dims (optional)
            action_mean = action_mean * available_actions

            # Also, you might want to force a small std for invalid dims so it’s basically “locked”
            # or you could keep them large—depends on your approach
            action_logstd = action_logstd * available_actions
        return FixedNormal(action_mean, action_logstd.exp(),self.action_scale,self.action_bias)


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


class MultiVariate(nn.Module):
    def __init__(self, state_dim, action_dim, action_space=None):
        hidden_dim=state_dim
        super(MultiVariate, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        if action_space is None:
            self.action_scale = torch.nn.Parameter(torch.tensor(1.),requires_grad=False)
            self.action_bias = torch.nn.Parameter(torch.tensor(0.),requires_grad=False)
        else:
            self.action_scale = torch.nn.Parameter(torch.FloatTensor(
                (action_space.high - action_space.low) / 2.),requires_grad=False)
            self.action_bias = torch.nn.Parameter(torch.FloatTensor(
                (action_space.high + action_space.low) / 2.),requires_grad=False)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        cov_matrix = torch.diag_embed(std ** 2)  # Diagonal covariance matrix
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov_matrix)

        # normal = torch.distributions.Normal(mean, std)
        # Sampling an action
        raw_action = dist.rsample()  # Reparameterization trick
        action = torch.tanh(raw_action) * self.action_scale + self.action_bias
        # Log probability of the action
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.sum(torch.log(self.action_scale * (1 - torch.tanh(raw_action).pow(2)) + 1e-6), dim=-1)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        # Entropy of the distribution
        entropy = dist.entropy().sum(dim= -1).mean()

        return action, log_prob, entropy


    def evaluate_action(self, state, action):
        """
        Evaluates the log probability and entropy of the given action for a given state.

        Args:
            state (torch.Tensor): The input state.
            action (torch.Tensor): The action to evaluate.

        Returns:
            log_prob (torch.Tensor): The log probability of the action.
            entropy (torch.Tensor): The entropy of the distribution.
        """
        mean, std = self.forward(state)
        cov_matrix = torch.diag_embed(std ** 2)  # Diagonal covariance matrix
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov_matrix)

        # Compute log probability
        raw_action = torch.atanh((action - self.action_bias) / self.action_scale)
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.sum(torch.log(self.action_scale * (1 - torch.tanh(raw_action).pow(2)) + 1e-6), dim=-1)

        # Compute entropy
        entropy = dist.entropy().sum(dim= -1).mean()

        return log_prob, entropy


# class NormalDist(nn.Module):
#     def __init__(self, state_dim, action_dim, action_space=None):
#         hidden_dim=state_dim
#         super(NormalDist, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.mean = nn.Linear(hidden_dim, action_dim)
#         # self.log_std = nn.Linear(hidden_dim, action_dim)
#         action_std_init=0.5
#         self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * action_std_init)
#         if action_space is None:
#             self.action_scale = torch.nn.Parameter(torch.tensor(1.),requires_grad=False)
#             self.action_bias = torch.nn.Parameter(torch.tensor(0.),requires_grad=False)
#         else:
#             self.action_scale = torch.nn.Parameter(torch.FloatTensor(
#                 (action_space.high - action_space.low) / 2.),requires_grad=False)
#             self.action_bias = torch.nn.Parameter(torch.FloatTensor(
#                 (action_space.high + action_space.low) / 2.),requires_grad=False)
#
#
#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         mean = self.mean(x)
#         # log_std = self.log_std(x).clamp(-20, 2)
#         # std = log_std.exp()
#         action_std = torch.exp(self.action_log_std)
#         return mean, action_std
#
#     def sample(self, state):
#         mean, std = self.forward(state)
#         dist = torch.distributions.Normal(mean, std)
#         x_t = dist.rsample()
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#         log_prob = dist.log_prob(x_t)
#         # Enforcing Action Bound
#         log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
#         log_prob = log_prob.sum(1, keepdim=True)
#         entropy = dist.entropy()
#
#         return action, log_prob, entropy
#
#
#     def evaluate_action(self, state, action):
#         """
#         Evaluates the log probability and entropy of the given action for a given state.
#
#         Args:
#             state (torch.Tensor): The input state.
#             action (torch.Tensor): The action to evaluate.
#
#         Returns:
#             log_prob (torch.Tensor): The log probability of the action.
#             entropy (torch.Tensor): The entropy of the distribution.
#         """
#
#         mean, std = self.forward(state)
#         if torch.any(torch.isnan(mean)):
#             print()
#         if torch.any(torch.isnan(std)):
#             print()
#         dist = torch.distributions.Normal(mean, std)
#
#         # Compute log probability
#         raw_action = torch.atanh((action - self.action_bias) / self.action_scale)
#         log_prob = dist.log_prob(raw_action)
#         log_prob -= torch.sum(torch.log(self.action_scale * (1 - torch.tanh(raw_action).pow(2)) + 1e-6), dim=-1,keepdim=True)
#
#         # Compute entropy
#         entropy = dist.entropy()
#
#         return log_prob, entropy

class NormalDist(nn.Module):
    def __init__(self, state_dim, action_dim, action_space=None):
        hidden_dim = state_dim
        super(NormalDist, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        if action_space is None:
            self.action_scale = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
            self.action_bias = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)
        else:
            self.action_scale = torch.nn.Parameter(torch.FloatTensor(
                (action_space.high - action_space.low) / 2.), requires_grad=False)
            self.action_bias = torch.nn.Parameter(torch.FloatTensor(
                (action_space.high + action_space.low) / 2.), requires_grad=False)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)

        # Sampling an action
        raw_action = dist.rsample()  # Reparameterization trick
        action = torch.tanh(raw_action) * self.action_scale + self.action_bias
        # Log probability of the action
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.sum(torch.log(self.action_scale * (1 - torch.tanh(raw_action).pow(2)) + 1e-6), dim=-1)
        # Entropy of the distribution
        entropy = dist.entropy().sum(dim= -1).mean()

        return action, log_prob, entropy

    def evaluate_action(self, state, action):
        """
        Evaluates the log probability and entropy of the given action for a given state.

        Args:
            state (torch.Tensor): The input state.
            action (torch.Tensor): The action to evaluate.

        Returns:
            log_prob (torch.Tensor): The log probability of the action.
            entropy (torch.Tensor): The entropy of the distribution.
        """
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)

        # Compute log probability
        raw_action = torch.atanh((action - self.action_bias) / self.action_scale)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.sum(torch.log(self.action_scale * (1 - torch.tanh(raw_action).pow(2)) + 1e-6), dim=-1)

        # Compute entropy
        entropy = dist.entropy().sum(dim=-1).mean()

        return log_prob, entropy