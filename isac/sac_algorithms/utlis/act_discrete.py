import numpy as np
from .distributions import Categorical
import torch
import torch.nn as nn


class DiscreteACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """

    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, tpdv):
        super(DiscreteACTLayer, self).__init__()
        self.tpdv = tpdv
        action_dim = action_space.n
        self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)

    def forward(self, x, available_actions=None, deterministic=False):
        action_logits = self.action_out(x)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs, available_actions

    def get_probs(self, x, available_actions=None):
        action_logits = self.action_out(x)
        action_probs = action_logits.probs
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None):
            action_logits = self.action_out(x)
            action_log_probs = action_logits.log_probs(action)
            dist_entropy = action_logits.entropy().mean()
            return action_log_probs, dist_entropy
