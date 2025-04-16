import torch
import torch.optim as optim
from copy import deepcopy
from isac.sac_algorithms.sac_actor_critic import SACActor, DistributionalSACCritic
from isac.sac_utils.util import update_linear_schedule


class SACPolicy:
    def __init__(self, args, obs_space, state_space, act_space,num_agents, device=torch.device("cpu"), metadata: dict = None):
        self.device = device
        # The actor uses local observations (obs_space)
        self.actor = SACActor(args, obs_space, act_space, device, metadata=metadata)
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr

        # The critic now uses the global state (state_space) along with joint actions.
        self.critic = DistributionalSACCritic(args, state_space, act_space,num_agents, device)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=args.actor_lr,
                                                eps=args.opti_eps,
                                                weight_decay=args.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=args.critic_lr,
                                                 eps=args.opti_eps,
                                                 weight_decay=args.weight_decay)
        self.act_space = act_space

    def lr_decay(self, episode, episodes):
        actor_lr = update_linear_schedule(self.actor_optimizer, episode, episodes, self.actor_lr)
        critic_lr = update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        return actor_lr, critic_lr

    def get_actions(self, obs, available_actions=None, deterministic=False):
        actions, log_pi, available_actions = self.actor(obs, available_actions, deterministic)
        return actions, log_pi

    def evaluate_actions(self, obs, action, available_actions=None):
        _, log_pi, _ = self.actor(obs, available_actions, deterministic=False)
        return log_pi
