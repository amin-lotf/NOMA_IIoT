import torch
from tensorflow.python.keras.backend import epsilon
from torch import nn

from duel_ddqn.duel_ddqn_algorithms.r_madqn.algorithm.duel_ddqn_actor import MADQNActor
from duel_ddqn.duel_ddqn_utils.util import update_linear_schedule


class DuelDDQNPolicy:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu"),metadata:dict=None):
        self.device = device
        self.lr = args.lr
        self.epsilon=args.dqn_epsilon_start
        self.args=args
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.obs_space = obs_space
        self.act_space = act_space
        self.actor = MADQNActor(args, self.obs_space, self.act_space,self.device,metadata=metadata)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.loss_func = nn.MSELoss()

    def lr_decay(self, episode, episodes):
        return update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)

    def epsilon_decay(self,episode):
        self.epsilon = max(self.args.dqn_epsilon_end, self.args.dqn_epsilon_start -
                      episode / self.args.dqn_epsilon_decay_last_episode)

    def get_actions(self, obs, available_actions ):
       # eps = 0 if deterministic else self.epsilon
       actions= self.actor(obs,available_actions,self.epsilon)
       return actions

    # def get_vals(self,obs):
    #     actions_vals = self.actor.get_vals(obs)
    #     return actions_vals

    def get_batch_acts(self,obs,available_actions):
        actions = self.actor.get_acts(obs,available_actions)
        return actions

    def evaluate_actions(self, obs, action,available_actions):
        action_vals = self.actor.evaluate_actions(obs,action,available_actions)
        return action_vals
