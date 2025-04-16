import numpy as np

from .distributions import Categorical, DiagGaussian, MultiVariate, NormalDist
import torch
import torch.nn as nn



class ContinuousACTLayer(nn.Module):
    def __init__(self, action_space, inputs_dim,metadata:dict=None):
        super(ContinuousACTLayer, self).__init__()
        action_dim = action_space.shape[0]
        self.act_min=action_space.low[0]
        self.act_max=action_space.high[0]
        # self.action_out = MultiVariate(inputs_dim, action_dim, action_space)
        # self.action_out = NormalDist(inputs_dim, action_dim, action_space)
        self.action_out = DiagGaussian(inputs_dim, action_dim,action_space)
        self.maddpg = True
        self.noise_scale = 0.0 # for continuous actions
        self.noise_epsilon = 0.0  # for discrete actions

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
            return torch.clamp(action + torch.randn_like(action) * self.noise_scale, self.act_min, self.act_max)
            # return action + torch.randn_like(action) * self.noise_scale
        else:
            return action

    def forward(self, x, available_actions=None, noise_scale=None, noise_epsilon=None, deterministic=None):
        if noise_scale:
            self.noise_scale = noise_scale
        if noise_epsilon:
            self.noise_epsilon = noise_epsilon
        # actions, action_log_probs,_=self.action_out.sample(x)
        action_logits = self.action_out(x)
        # actions = action_logits.mode() if deterministic else action_logits.sample()
        actions =  action_logits.sample()
        if self.maddpg and not deterministic:
            actions = self.add_maddpg_noise(actions, action_logits, type='Normal')
        action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs,available_actions


    #     return action_probs

    def evaluate_actions(self, x, action,available_actions=None):
        # log_prob, entropy = self.action_out.evaluate_action(x,action)
        action_logits = self.action_out(x)
        log_prob = action_logits.log_probs(action)
        entropy = action_logits.entropy().mean()
        return  log_prob, entropy