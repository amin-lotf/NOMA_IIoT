import numpy as np

from .distributions import Categorical, DiagGaussian, MultiVariate, NormalDist
import torch
import torch.nn as nn

from ...ppo_utils.util import check


class ContinuousACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim,metadata:dict=None):
        super(ContinuousACTLayer, self).__init__()
        action_dim = action_space.shape[0]
        # self.action_out = MultiVariate(inputs_dim, action_dim, action_space)
        # self.action_out = NormalDist(inputs_dim, action_dim, action_space)
        self.action_out = DiagGaussian(inputs_dim, action_dim,action_space)




    def forward(self, x,available_actions=None, deterministic=None):
        # actions, action_log_probs,_=self.action_out.sample(x)
        action_logits = self.action_out(x,available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        if available_actions is not None:
            actions = actions * available_actions
        action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs,available_actions


    #     return action_probs

    def evaluate_actions(self, x, action,available_actions=None):
        # log_prob, entropy = self.action_out.evaluate_action(x,action)
        action_logits = self.action_out(x,available_actions)
        log_prob = action_logits.log_probs(action)
        entropy = action_logits.entropy().mean()
        return  log_prob, entropy