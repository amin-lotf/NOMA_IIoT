import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from isac.sac_algorithms.utlis.act_continuous import ContinuousACTLayer
from isac.sac_algorithms.utlis.act_discrete import DiscreteACTLayer
from isac.sac_algorithms.utlis.act_mixed import MixedACTLayer
from isac.sac_algorithms.utlis.act_multi_discrete import MultiDiscreteACTLayer
from isac.sac_algorithms.utlis.mlp import MLPBase
from isac.sac_algorithms.utlis.resnet_v2 import ResNet
from isac.sac_algorithms.utlis.util import get_clones, init
from isac.sac_utils.util import get_shape_from_obs_space, get_shape_from_act_space


# from ppo.algorithms.utils.act_ppo_continuous import ContinuousACTLayer
# from ppo.algorithms.utils.act_ppo_discrete import DiscreteACTLayer
# from ppo.algorithms.utils.act_ppo_multi_discrete import MultiDiscreteACTLayer
# from ppo.algorithms.utils.act_ppo_mixed import MixedACTLayer
# from ppo.algorithms.utils.resnet_v2 import ResNet
# from ppo.algorithms.utils.mlp import MLPBase
# from ppo.ppo_utils.util import get_shape_from_obs_space, get_shape_from_act_space


#####################################
# SAC Actor (Policy) Network
#####################################

class SACActor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu"), metadata: dict = None):
        super(SACActor, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self.use_resnet = args.use_resnet
        self._use_orthogonal = args.use_orthogonal
        self._use_feature_normalization = args.use_feature_normalization
        self._layer_n = args.layer_N
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), self._gain)

        self.tpdv = dict(dtype=torch.float32, device=device)
        obs_shape = get_shape_from_obs_space(obs_space)
        self.layer1 = init_(nn.Linear(obs_shape[0], args.hidden_size))
        # self.layer2 = nn.Linear(args.hidden_size, args.hidden_size)
        fc_h = nn.Sequential(init_(nn.Linear(args.hidden_size, args.hidden_size)))
        self.layer2 = get_clones(fc_h, self._layer_n)

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_shape[0])

        if action_space.__class__.__name__ == "Discrete":
            self.act = DiscreteACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, self.tpdv)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.act = MultiDiscreteACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, device,
                                             metadata=metadata)
        elif action_space.__class__.__name__ == "Box":
            self.act = ContinuousACTLayer(action_space, self.hidden_size)
        else:
            self.act = MixedACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, device,
                                     metadata=metadata)
        self.to(device)

    def forward(self, obs, available_actions=None, deterministic=False):
        if self._use_feature_normalization:
            obs = self.feature_norm(obs)
        features = torch.relu(self.layer1(obs))
        for i in range(self._layer_n):
            features = torch.relu(self.layer2[i](features))
        actions, action_log_probs, available_actions = self.act(features, available_actions, deterministic)
        return actions, action_log_probs, available_actions


#####################################
# Distributional SAC Critic Network
#####################################

class DistributionalSACCritic(nn.Module):
    def __init__(self, args, state_space, action_space, num_agents,device=torch.device("cpu")):
        super(DistributionalSACCritic, self).__init__()
        self.hidden_size = args.hidden_size
        self.use_resnet = args.use_resnet
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self._layer_n = args.layer_N
        self._use_feature_normalization=args.use_feature_normalization
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), self._gain)
        # Distributional parameters
        self.num_atoms = args.num_atoms
        self.Vmin = args.v_min
        self.Vmax = args.v_max
        self.delta_z = (self.Vmax - self.Vmin) / (self.num_atoms - 1)
        self.register_buffer('support', torch.linspace(self.Vmin, self.Vmax, self.num_atoms))

        # Global state shape (e.g., [state_dim]) and joint actions flattened.
        state_shape = state_space.shape  # state_space is the global state space.
        act_shape = get_shape_from_act_space(action_space)
        input_dim = state_shape[0] + int(np.prod(act_shape))*num_agents
        self.q_1_layer1 = init_(nn.Linear(input_dim, args.hidden_size))
        q_1_fc_h = nn.Sequential(init_(nn.Linear(args.hidden_size, args.hidden_size)))
        self.q_1_layer2 = get_clones(q_1_fc_h, self._layer_n)

        self.q_2_layer1 = init_(nn.Linear(input_dim, args.hidden_size))
        q_2_fc_h = nn.Sequential(init_(nn.Linear(args.hidden_size, args.hidden_size)))
        self.q_2_layer2 = get_clones(q_2_fc_h, self._layer_n)

        self.q1_out = init_(nn.Linear(self.hidden_size, self.num_atoms))
        self.q2_out = init_(nn.Linear(self.hidden_size, self.num_atoms))
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)
        self.to(device)

    def forward(self, state, action):
        # If actions come as (batch, n_agents, act_dim), flatten to (batch, n_agents * act_dim)
        if len(action.shape) > 2:
            action = action.view(action.shape[0], -1)
        x = torch.cat([state, action], dim=-1)
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        q1_features = torch.relu(self.q_1_layer1(x))
        for i in range(self._layer_n):
            q1_features = torch.relu(self.q_1_layer2[i](q1_features))
        q2_features = torch.relu(self.q_2_layer1(x))
        for i in range(self._layer_n):
            q2_features = torch.relu(self.q_2_layer2[i](q2_features))
        q1_logits = self.q1_out(q1_features)
        q2_logits = self.q2_out(q2_features)
        q1_dist = F.softmax(q1_logits, dim=-1)
        q2_dist = F.softmax(q2_logits, dim=-1)
        return q1_dist, q2_dist

    def get_expectation(self, dist):
        return torch.sum(dist * self.support, dim=1, keepdim=True)
