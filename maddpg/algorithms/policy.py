from copy import deepcopy

import numpy as np

from maddpg.algorithms.actor_critic import VectorizedMADDPGActor, CentralizedMADDQNCritic
import torch

from maddpg.maddpg_utils.util import update_linear_schedule


class VectorizedAttentionMADDPGPolicy:
    def __init__(self, args, num_agents, obs_space, state_space, act_space, device=torch.device("cpu"), metadata: dict = None):
        self.device = device
        self.num_agents = num_agents
        self.actor_lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.gamma = args.gamma
        self.tau = args.tau
        self.noise_epsilon=args.epsilon_start
        self.noise_scale=args.epsilon_start
        self.epsilon_start=args.epsilon_start
        self.epsilon_end=args.epsilon_end
        self.decay_steps=args.epsilon_decay_last_episode
        self.decay_rate=args.decay_rate
        self.warmup_steps=args.start_train_step
        # Use a single shared actor for all agents (vectorized across batch and agent dimensions).
        self.actor = VectorizedMADDPGActor(args, obs_space, act_space, device, metadata)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.actor_lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        # The critic remains unchanged (using our attention-based critic).
        self.critic = CentralizedMADDQNCritic(args, state_space,act_space,num_agents, device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

    # def epsilon_decay(self,episode):
    #     self.noise_epsilon = max(self.epsilon_end, self.epsilon_start -
    #                   episode / self.epsilon_decay_last_episode)
    #     self.noise_scale = max(self.epsilon_end, self.epsilon_start -
    #                              episode / self.epsilon_decay_last_episode)


    # def epsilon_decay(self, episode):
    #     # Define a decay rate hyperparameter. Adjust this value based on your task.
    #     # self.decay_rate = 100.0  # example value; you might want to tune it
    #     self.noise_epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-episode / self.decay_rate)
    #     self.noise_scale = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-episode / self.decay_rate)

    def lr_decay(self, episode, episodes):
        actor_lr = update_linear_schedule(self.actor_optimizer, episode, episodes, self.actor_lr)
        critic_lr = update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        return actor_lr, critic_lr

    def epsilon_decay(self,step_index):
        if step_index < self.warmup_steps:
            return self.epsilon_start
        else:
            effective_step = step_index - self.warmup_steps
            # Exponential decay: noise decays slowly over 'decay_steps'
            self.noise_epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(- effective_step / self.decay_steps)
            self.noise_scale = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(- effective_step / self.decay_steps)


    def act(self, obs,  available_actions=None,deterministic=False):
        # # obs: (B, N, obs_dim)
        # B, N, _ = obs.shape
        # obs_flat = obs.view(B * N, -1)  # Flatten to (B*N, obs_dim)
        actions = self.actor(obs, available_actions,self.noise_scale,self.noise_epsilon, deterministic)
        # actions = actions_flat.view(B, N, -1)  # Reshape back to (B, N, act_dim)
        return actions

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()
