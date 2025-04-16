import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ppo.algorithms.utils.act_ppo_continuous import ContinuousACTLayer
from ppo.algorithms.utils.act_ppo_discrete import DiscreteACTLayer
from ppo.algorithms.utils.act_ppo_mixed import MixedACTLayer
from ppo.algorithms.utils.act_ppo_multi_discrete import MultiDiscreteACTLayer
from ppo.algorithms.utils.distributions import DiagGaussian
from ppo.algorithms.utils.mlp import MLPBase
from isac.sac_utils.sac_buffer import SACBuffer
from isac.sac_utils.util import get_shape_from_obs_space, get_shape_from_act_space


# Neural Networks
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Actor(nn.Module):
    def __init__(self, args, obs_space, action_space,device=torch.device("cpu"),metadata:dict=None):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)
        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = MLPBase(args, obs_shape[0])

        if action_space.__class__.__name__ == "Discrete":
            self.act = DiscreteACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain,self.tpdv)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.act = MultiDiscreteACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain,self.tpdv,metadata=metadata)
        elif action_space.__class__.__name__ == "Box":
            self.act = ContinuousACTLayer(action_space, self.hidden_size)
        else:
            self.act = MixedACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, device, metadata=metadata)
        self.to(device)

    def forward(self, obs,  available_actions=None,deterministic=False):
        actor_features = self.base(obs)
        actions, action_log_probs,available_actions = self.act(actor_features, available_actions, deterministic)
        return actions, action_log_probs

    def get_state_dict(self):
        # return self.hidden_layer.state_dict()
        return self.state_dict()

    def evaluate_actions(self, obs,  action, available_actions=None):
        actor_features = self.base(obs)
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,action, available_actions)
        return action_log_probs, dist_entropy



# SAC Agent
class SACAgent:
    def __init__(self, args,observation_space, action_space, buffer:SACBuffer,device,metadata=None ):
        self.device=device
        self.action_space = action_space
        self.replay_buffer = buffer
        self.gamma = args.gamma
        self.tau = args.tau
        state_dim = observation_space.shape[0]
        action_dim=get_shape_from_act_space(action_space)

        self.policy_net = Actor(args, observation_space, action_space, self.device,metadata=metadata)


        self.q_net1 = SoftQNetwork(state_dim, action_dim, args.hidden_size).to(device)
        self.q_net2 = SoftQNetwork(state_dim, action_dim, args.hidden_size).to(device)
        self.target_q_net1 = SoftQNetwork(state_dim, action_dim, args.hidden_size).to(device)
        self.target_q_net2 = SoftQNetwork(state_dim, action_dim, args.hidden_size).to(device)

        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=args.lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=args.lr)

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.lr_alpha)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state,available_actions=None):
        actions,_ = self.policy_net(state.unsqueeze(0),available_actions)
        return actions

    def update(self):
        state, next_state,action, reward , done,available_actions = self.replay_buffer.feed_forward_generator()


        with torch.no_grad():
            next_action,next_log_prob = self.policy_net(next_state,available_actions)
            target_q1 = self.target_q_net1(next_state, next_action)
            target_q2 = self.target_q_net2(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * (torch.min(target_q1, target_q2) - self.alpha * next_log_prob)

        current_q1 = self.q_net1(state, action)
        current_q2 = self.q_net2(state, action)
        q_loss1 = F.mse_loss(current_q1, target_q)
        q_loss2 = F.mse_loss(current_q2, target_q)

        self.q_optimizer1.zero_grad()
        q_loss1.backward()
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q_loss2.backward()
        self.q_optimizer2.step()

        new_action,log_prob = self.policy_net(state, available_actions)
        q1_new = self.q_net1(state, new_action)
        q2_new = self.q_net2(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_prob - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)