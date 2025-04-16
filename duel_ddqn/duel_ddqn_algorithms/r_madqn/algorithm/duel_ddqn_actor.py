import torch
import torch.nn as nn

from duel_ddqn.duel_ddqn_algorithms.r_madqn.algorithm.utlis.act_discrete import DiscreteACTLayer
from duel_ddqn.duel_ddqn_algorithms.r_madqn.algorithm.utlis.act_multi_discrete import MultiDiscreteDQNActLayer

from duel_ddqn.duel_ddqn_algorithms.r_madqn.algorithm.utlis.mlp import MLPBase
from duel_ddqn.duel_ddqn_algorithms.r_madqn.algorithm.utlis.resnet_v2 import ResNet
from duel_ddqn.duel_ddqn_algorithms.r_madqn.algorithm.utlis.util import check, get_clones, init
from duel_ddqn.duel_ddqn_utils.util import get_shape_from_obs_space


class MADQNActor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu"), metadata: dict = None):
        super(MADQNActor, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self.use_resnet = args.use_resnet
        self._layer_n = args.layer_N
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self._use_feature_normalization = args.use_feature_normalization
        # self.base = MLPBase(args, obs_shape[0]).float()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), self._gain)
        self.device=device
        obs_shape = get_shape_from_obs_space(obs_space)
        self.layer1 = init_(nn.Linear(obs_shape[0], args.hidden_size))
        # self.layer2 = nn.Linear(args.hidden_size, args.hidden_size)
        fc_h = nn.Sequential(init_(nn.Linear(args.hidden_size, args.hidden_size)))
        self.layer2 = get_clones(fc_h, self._layer_n)
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_shape)
        # For SAC we assume a feed-forward (nonrecurrent) actor.
        # self.base = base(obs_shape[0], self.hidden_size, args.resnet_blocks) if self.use_resnet else base(args,
        #                                                                                                   obs_shape[0])

        self.act = MultiDiscreteDQNActLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, device,args=args,
                                         metadata=metadata)
        self.to(device)

    def forward(self, obs,available_actions, epsilon):
        if self._use_feature_normalization:
            obs = self.feature_norm(obs)
        features = torch.relu(self.layer1(obs))
        for i in range(self._layer_n):
            features = torch.relu(self.layer2[i](features))
        actions = self.act(features, epsilon,available_actions)
        return actions

    def evaluate_actions(self, obs, action,available_actions):
        if self._use_feature_normalization:
            obs = self.feature_norm(obs)
        features = torch.relu(self.layer1(obs))
        for i in range(self._layer_n):
            features = torch.relu(self.layer2[i](features))
        action_vals = self.act.evaluate_actions(features, action,available_actions)
        return action_vals


    def get_acts(self, obs,available_actions):
        if self._use_feature_normalization:
            obs = self.feature_norm(obs)
        features = torch.relu(self.layer1(obs))
        for i in range(self._layer_n):
            features = torch.relu(self.layer2[i](features))
        epsilon=0.0
        actions = self.act(features,epsilon ,available_actions)
        return actions
