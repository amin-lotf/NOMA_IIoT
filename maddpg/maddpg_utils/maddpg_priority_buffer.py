import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from maddpg.maddpg_utils.util import get_shape_from_obs_space, get_shape_from_act_space

# === Replay Buffer remains largely unchanged ===
# (You can continue storing per-agent observations and actions;
# later the trainer will form the global state by concatenating
# the agents’ observations.)

class MADDPGPriorityReplayBuffer(object):
    def __init__(self, args, n_agents, obs_space, state_space, act_space, available_act_shape, device, shift_reward=0):
        self.buffer_length = args.buffer_length + shift_reward
        self.batch_size = args.batch_size
        self.shift_reward = shift_reward
        self.hidden_size = args.hidden_size
        self.buffer_device = device
        # self.buffer_device = torch.device('cpu')
        self.batch_device = device

        self.n_envs = args.n_envs
        self.n_agents = n_agents

        obs_shape = get_shape_from_obs_space(obs_space)
        if isinstance(obs_shape[-1], list):
            obs_shape = obs_shape[:1]
        self.obs = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, *obs_shape),
                               dtype=torch.float32, device=self.buffer_device)
        self.next_obs = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, *obs_shape),
                                    dtype=torch.float32, device=self.buffer_device)

        state_shape = get_shape_from_obs_space(state_space)
        if isinstance(state_shape[-1], list):
            state_shape = state_shape[:1]
        self.state = torch.zeros((self.buffer_length, self.n_envs, *state_shape),
                                 dtype=torch.float32, device=self.buffer_device)
        self.next_state = torch.zeros((self.buffer_length, self.n_envs, *state_shape),
                                      dtype=torch.float32, device=self.buffer_device)

        # (Global state can be obtained by concatenating agent obs.)
        # Actions:
        act_shape = get_shape_from_act_space(act_space)
        act_space_type = act_space.__class__.__name__
        if act_space_type == 'Discrete' or act_space_type == 'MultiDiscrete':
            self.actions = np.zeros((self.buffer_length, self.n_envs, self.n_agents, act_shape), dtype=np.int64)
            if type(available_act_shape) is tuple:
                self.available_actions = torch.ones(
                    (self.buffer_length, self.n_envs, self.n_agents, *available_act_shape),
                    dtype=torch.int32, device=self.buffer_device)
            else:
                self.available_actions = torch.ones(
                    (self.buffer_length, self.n_envs, self.n_agents, available_act_shape),
                    dtype=torch.int32, device=self.buffer_device)
        elif act_space_type == 'Box':
            self.actions = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, act_shape), dtype=torch.float32,
                                       device=self.buffer_device)
            self.available_actions = torch.ones((self.buffer_length, self.n_envs, self.n_agents, available_act_shape),
                                                dtype=torch.int32, device=self.buffer_device)
        elif act_space_type == 'list':
            self.actions = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, act_shape), dtype=torch.float32,
                                       device=self.buffer_device)
            if type(available_act_shape) is tuple:
                self.available_actions = torch.ones(
                    (self.buffer_length, self.n_envs, self.n_agents, *available_act_shape),
                    dtype=torch.int32, device=self.buffer_device)
            else:
                self.available_actions = torch.ones(
                    (self.buffer_length, self.n_envs, self.n_agents, available_act_shape),
                    dtype=torch.int32, device=self.buffer_device)
        else:
            raise NotImplementedError('Action space type not supported!')

        self.next_available_actions = torch.ones_like(self.available_actions)
        self.rewards = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, 1), dtype=torch.float32,
                                   device=self.buffer_device)
        self.masks = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, 1),
                                 dtype=torch.float32, device=self.buffer_device)
        self.priorities = torch.ones((self.buffer_length, self.n_envs), dtype=torch.float32, device=self.buffer_device)
        self.priority_exponent = getattr(args, "priority_exponent", 1.0)
        self.step = 0

    def __len__(self):
        return min(self.step, self.buffer_length)

    def insert(self, data):
        obs, next_obs, state, next_state, actions, rewards, masks, available_actions, next_available_actions = data
        step = self.step % self.buffer_length
        if torch.is_tensor(obs):
            self.obs[step].copy_(obs.to(self.buffer_device))
        else:
            self.obs[step] = torch.as_tensor(obs).to(self.buffer_device).float()
        if torch.is_tensor(next_obs):
            self.next_obs[step].copy_(next_obs.to(self.buffer_device))
        else:
            self.next_obs[step] = torch.as_tensor(next_obs).to(self.buffer_device).float()

        if torch.is_tensor(state):
            self.state[step].copy_(state.to(self.buffer_device))
        else:
            self.state[step] = torch.as_tensor(state).to(self.buffer_device).float()

        if torch.is_tensor(next_state):
            self.next_state[step].copy_(next_state.to(self.buffer_device))
        else:
            self.next_state[step] = torch.as_tensor(next_state).to(self.buffer_device).float()

        if torch.is_tensor(actions):
            self.actions[step].copy_(actions.to(self.buffer_device))
        else:
            self.actions[step] = torch.as_tensor(actions).to(self.buffer_device).float()

        if available_actions is not None:
            if torch.is_tensor(available_actions):
                self.available_actions[step].copy_(available_actions.to(self.buffer_device))
            else:
                self.available_actions[step].copy_(torch.as_tensor(available_actions).to(self.buffer_device))
        if next_available_actions is not None:
            if torch.is_tensor(next_available_actions):
                self.next_available_actions[step].copy_(next_available_actions.to(self.buffer_device))
            else:
                self.next_available_actions[step].copy_(torch.as_tensor(next_available_actions).to(self.buffer_device))
        if torch.is_tensor(rewards):
            self.rewards[step - self.shift_reward].copy_(rewards.to(self.buffer_device))
        else:
            self.rewards[step - self.shift_reward].copy_(torch.as_tensor(rewards).to(self.buffer_device))
        self.masks[step].copy_(torch.as_tensor(masks).to(self.buffer_device))
        current_max = self.priorities.max() if self.step > 0 else 1.0
        self.priorities[step].fill_(current_max)
        self.step += 1

    def sample(self):
        T = self.batch_size
        N = self.n_envs
        A = self.n_agents
        batch_size = T * N
        active_length = len(self)
        if batch_size > active_length:
            raise ValueError('Batch size must be smaller than current buffer length')
        step = self.step % self.buffer_length
        to_be_removed = torch.arange(step - self.shift_reward, step, device=self.buffer_device)
        current_indices = torch.arange(active_length, device=self.buffer_device)
        mask = torch.ones_like(current_indices)
        mask[to_be_removed] = 0
        current_indices = current_indices[mask == 1]

        flat_priorities = self.priorities[current_indices].reshape(-1)
        sorted_priorities, sorted_indices = torch.sort(flat_priorities, descending=True)
        ranks = torch.empty_like(sorted_indices, dtype=torch.float32)
        ranks[sorted_indices] = torch.arange(1, sorted_indices.numel() + 1, device=flat_priorities.device,
                                             dtype=torch.float32)
        p_e = (1.0 / ranks) ** self.priority_exponent
        probs = p_e / p_e.sum()
        sampled_flat_indices = torch.multinomial(probs, batch_size, replacement=False)

        obs = self.obs[current_indices].reshape(-1, A, *self.obs.shape[3:])
        next_obs = self.next_obs[current_indices].reshape(-1, A, *self.next_obs.shape[3:])
        state = self.state[current_indices].reshape(-1,  *self.state.shape[2:])
        next_state = self.next_state[current_indices].reshape(-1,  *self.next_state.shape[2:])
        actions = self.actions[current_indices].reshape(-1, A, *self.actions.shape[3:])
        rewards = self.rewards[current_indices].reshape(-1, A, *self.rewards.shape[3:])
        masks = self.masks[current_indices].reshape(-1, A, *self.masks.shape[3:])
        available_actions = self.available_actions[current_indices].reshape(-1, A, *self.available_actions.shape[3:])
        next_available_actions = self.next_available_actions[current_indices].reshape(-1, A, *self.next_available_actions.shape[3:])
        obs = obs[sampled_flat_indices].to(self.batch_device, non_blocking=True)
        next_obs = next_obs[sampled_flat_indices].to(self.batch_device, non_blocking=True)
        state = state[sampled_flat_indices].to(self.batch_device, non_blocking=True)
        next_state = next_state[sampled_flat_indices].to(self.batch_device, non_blocking=True)
        actions = actions[sampled_flat_indices].to(self.batch_device, non_blocking=True)
        rewards = rewards[sampled_flat_indices].to(self.batch_device, non_blocking=True).squeeze(-1)
        masks = masks[sampled_flat_indices].to(self.batch_device, non_blocking=True).squeeze(-1)
        available_actions = available_actions[sampled_flat_indices].to(self.batch_device, non_blocking=True)
        next_available_actions = next_available_actions[sampled_flat_indices].to(self.batch_device, non_blocking=True)

        return obs, next_obs,state,next_state, actions, rewards, masks, available_actions, next_available_actions, sampled_flat_indices

    def update_priorities(self, flat_indices, new_priorities):
        current_length = min(self.step, self.buffer_length)
        time_indices = flat_indices // self.n_envs
        env_indices = flat_indices % self.n_envs
        self.priorities[:current_length][time_indices, env_indices] = new_priorities.to(self.buffer_device)

    def get_average_rewards(self, n):
        if self.step >= self.buffer_length:
            step = self.step % self.buffer_length
            inds = torch.arange(step - n, step)
        else:
            inds = torch.arange(max(0, self.step - n), self.step)
        return self.rewards[inds].mean()
