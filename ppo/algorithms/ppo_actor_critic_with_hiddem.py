from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from ppo.algorithms.utils.act_ppo_continuous import ContinuousACTLayer
from ppo.algorithms.utils.act_ppo_discrete import DiscreteACTLayer
from ppo.algorithms.utils.act_ppo_multi_discrete import MultiDiscreteACTLayer
from ppo.algorithms.utils.util import init, check
from ppo.algorithms.utils.cnn import CNNBase
from ppo.algorithms.utils.mlp import MLPBase
from ppo.algorithms.utils.rnn import RNNLayer
from ppo.ppo_utils.util import get_shape_from_obs_space

class HiddenFeature(nn.Module):
    def __init__(self,args, input_dim):
        super(HiddenFeature, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_n
        if args.use_cnn:
            self.base = CNNBase(args, input_dim)
        else:
            args_input = deepcopy(args)
            args_input.use_feature_normalization = False
            self.base = MLPBase(args_input, args_input.hidden_size)
        # base = CNNBase if  args.use_cnn else MLPBase
        # self.base = base(args, obs_shape)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

    def forward(self, features, rnn_states, masks):
        features = self.base(features)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            features, rnn_states = self.rnn(features, rnn_states, masks)
        return features, rnn_states



class Actor(nn.Module):
    def __init__(self, args, obs_space, action_space,device=torch.device("cpu"),metadata:dict=None):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_n
        self.tpdv = dict(dtype=torch.float32, device=device)
        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(obs_shape[0], self.hidden_size, self._recurrent_N, self._use_orthogonal)
            self.base = base(args, self.hidden_size)
        else:
            self.base = base(args, obs_shape[0])

        if action_space.__class__.__name__ == "Discrete":
            self.act = DiscreteACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain,self.tpdv)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.act = MultiDiscreteACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain,self.tpdv,metadata=metadata)
        elif action_space.__class__.__name__ == "Box":
            self.act = ContinuousACTLayer(action_space, self.hidden_size)
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        # input_features=self.input_layer(obs)
        # hidden_features,rnn_states=self.hidden_layer(input_features,rnn_states,masks)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            hidden_features, rnn_states = self.rnn(obs, rnn_states, masks)
            actor_features = self.base(hidden_features)
        else:
            actor_features = self.base(obs)
        actions, action_log_probs,available_actions = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs, rnn_states,available_actions

    def get_state_dict(self):
        # return self.hidden_layer.state_dict()
        return self.state_dict()

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None):
        input_features = self.input_layer(obs)
        hidden_features, rnn_states = self.hidden_layer(input_features, rnn_states, masks)
        # actor_features = self.base(obs)
        # if self._use_naive_recurrent_policy or self._use_recurrent_policy:
        #     hidden_features, rnn_states = self.rnn(hidden_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(hidden_features,action, available_actions)
        return action_log_probs, dist_entropy


class Critic(nn.Module):
    def __init__(self, args,obs, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_n
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        # base = CNNBase if args.use_cnn else MLPBase
        obs_shape = get_shape_from_obs_space(obs)
        # self.base = base(args, obs_shape)
        if args.use_cnn:
            self.base = CNNBase(args, obs_shape)
        else:
            args_input = deepcopy(args)
            args_input.layer_N = 0
            self.input_layer = MLPBase(args_input, obs_shape[0])
        self.hidden_layer = HiddenFeature(args, args.hidden_size)
        # if self._use_naive_recurrent_policy or self._use_recurrent_policy:
        #     self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def forward(self, obs, rnn_states, masks):
        input_features = self.input_layer(obs)
        hidden_features, rnn_states = self.hidden_layer(input_features, rnn_states, masks)
        values = self.v_out(hidden_features)
        return values, rnn_states

    def get_state_dict(self):
        # return self.hidden_layer.state_dict()
        return self.state_dict()
