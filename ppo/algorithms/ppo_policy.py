import torch
from ppo.algorithms.ppo_actor_critic import Actor, Critic
from ppo.ppo_utils.util import update_linear_schedule


class PPOPolicy:
    def __init__(self, args, obs_space,state_space, act_space,device=torch.device("cpu"),metadata:dict=None):
        self.device = device
        self.lr =  args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.act_space = act_space
        self.state_space=state_space
        self.actor = Actor(args, self.obs_space, self.act_space, self.device,metadata=metadata)
        self.critic = Critic(args,self.state_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, obs,state, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        actions, action_log_probs, rnn_states_actor,available_actions = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic( state,rnn_states_critic, masks)
        # actions=actions.item()
        return values, actions, action_log_probs, rnn_states_actor,available_actions, rnn_states_critic

    def get_values(self, state,rnn_states_critic, masks):
        values, _ = self.critic(state,rnn_states_critic, masks)
        return values

    def evaluate_actions(self, obs,state, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions
                                                                     )

        values, _ = self.critic(state,rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
