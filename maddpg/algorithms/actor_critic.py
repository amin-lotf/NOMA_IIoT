from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from maddpg.algorithms.utils.act_continuous import ContinuousACTLayer
from maddpg.algorithms.utils.act_discrete import DiscreteACTLayer
from maddpg.algorithms.utils.act_mixed import MixedACTLayer
from maddpg.algorithms.utils.act_multi_discrete import MultiDiscreteACTLayer
from maddpg.algorithms.utils.mlp import MLPBase
from maddpg.algorithms.utils.util import get_clones, init
from maddpg.maddpg_utils.util import get_shape_from_obs_space, get_shape_from_act_space


class VectorizedMADDPGActor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu"), metadata: dict = None):
        super(VectorizedMADDPGActor, self).__init__()
        self.hidden_size = args.hidden_size
        obs_shape = get_shape_from_obs_space(obs_space)
        self._layer_n=args.layer_N
        self._use_orthogonal=args.use_orthogonal
        self._gain = args.gain
        self._use_feature_normalization = args.use_feature_normalization
        # self.base = MLPBase(args, obs_shape[0]).float()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), self._gain)
        self.device = device
        self.layer1 = init_(nn.Linear(obs_shape[0], args.hidden_size))
        # self.layer2 = nn.Linear(args.hidden_size, args.hidden_size)
        fc_h = nn.Sequential(init_(nn.Linear(args.hidden_size, args.hidden_size)))
        self.layer2 = get_clones(fc_h, self._layer_n)
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_shape[0])

        # Instantiate the appropriate ACT layer based on action_space type.
        act_space_type = action_space.__class__.__name__
        if act_space_type == "Discrete":
            self.act = DiscreteACTLayer(action_space, self.hidden_size, args.use_orthogonal, args.gain,
                                        dict(dtype=torch.float32, device=device))
        elif act_space_type == "MultiDiscrete":
            self.act = MultiDiscreteACTLayer(action_space, self.hidden_size, args.use_orthogonal, args.gain, device, metadata)
        elif act_space_type == "Box":
            self.act = ContinuousACTLayer(action_space, self.hidden_size)
        elif isinstance(action_space, list):
            # If action_space is a list then we treat it as a mixed action space.
            self.act = MixedACTLayer(action_space, self.hidden_size, args.use_orthogonal, args.gain, device, metadata)
        else:
            # Fallback to mixed ACT layer if type is unknown.
            self.act = MixedACTLayer(action_space, self.hidden_size, args.use_orthogonal, args.gain, device, metadata)
        self.to(device)

    def forward(self, obs, available_actions=None,noise_scale=None,noise_epsilon=None, deterministic=False):
        # obs is expected to be of shape (B*N, obs_dim)
        # features = self.base(obs.float())  # (B*N, hidden_size)
        if self._use_feature_normalization:
            obs = self.feature_norm(obs)
        features = torch.relu(self.layer1(obs))
        for i in range(self._layer_n):
            features = torch.relu(self.layer2[i](features))
        # features = torch.relu(self.layer2(features))
        actions,_,_ = self.act(features, available_actions,noise_scale,noise_epsilon, deterministic)
        # Return dummy rnn state (None) for API compatibility.
        return actions




class CentralizedMADDQNCritic(nn.Module):
    def __init__(self, args, state_space, act_space, n_agents, device=torch.device("cpu")):
        super(CentralizedMADDQNCritic, self).__init__()
        state_shape = get_shape_from_obs_space(state_space)
        self._use_feature_normalization = args.use_feature_normalization
        self._layer_n=args.layer_N
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), self._gain)
        state_dim = state_shape[0]
        act_shape = get_shape_from_act_space(act_space)
        act_dim = act_shape  # assuming act_shape is an integer here
        input_dim = state_dim + n_agents * act_dim  # global state + flattened joint action
        self.fc1 = init_(nn.Linear(input_dim, args.hidden_size))
        fc_h = nn.Sequential(init_(nn.Linear(args.hidden_size, args.hidden_size)))
        self.layer2 = get_clones(fc_h, self._layer_n)
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)
        self.out = init_(nn.Linear(args.hidden_size, 1))
        self.to(device)

    def forward(self, state, actions):
        # state: (B, state_dim)  and actions: (B, n_agents*act_dim)
        x = torch.cat([state, actions], dim=-1)
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = F.relu(self.fc1(x))
        for i in range(self._layer_n):
            x = torch.relu(self.layer2[i](x))
        q = self.out(x)
        return q

