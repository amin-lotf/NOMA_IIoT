import torch
import numpy as np
from isac.sac_utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class PrioritizedSACBuffer(object):
    def __init__(self, args, n_agents, obs_space, state_space, act_space, available_act_shape, device, shift_reward=0):
        self.buffer_length = args.buffer_length
        self.batch_size = args.batch_size
        # self.device = torch.device('cpu')
        self.device = device
        self.training_device = device
        self.shift_reward = shift_reward
        self.n_envs = args.n_envs
        self.n_agents = n_agents

        # Local observations: still per agent.
        obs_shape = get_shape_from_obs_space(obs_space)
        self.obs = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, *obs_shape),
                               dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, *obs_shape),
                                    dtype=torch.float32, device=self.device)

        # Global state: stored once per environment (no agent dimension)
        state_shape = get_shape_from_obs_space(state_space)
        self.state = torch.zeros((self.buffer_length, self.n_envs, *state_shape),
                               dtype=torch.float32, device=self.device)
        self.next_state = torch.zeros((self.buffer_length, self.n_envs, *state_shape),
                                    dtype=torch.float32, device=self.device)

        # Rewards and masks: assumed to be global (shared) if using a centralized critic.
        self.rewards = torch.zeros((self.buffer_length, self.n_envs, 1), dtype=torch.float32, device=self.device)
        self.masks = torch.zeros((self.buffer_length, self.n_envs, 1), dtype=torch.int64, device=self.device)

        # Actions: still stored per agent.
        act_shape = get_shape_from_act_space(act_space)
        act_space_type = act_space.__class__.__name__
        if act_space_type == 'Discrete' or act_space_type == 'MultiDiscrete':
            self.actions = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, act_shape),
                                       dtype=torch.int64, device=self.device)
            self.available_actions = torch.ones(
                (self.buffer_length, self.n_envs, self.n_agents, available_act_shape),
                dtype=torch.int32, device=self.device)
            self.next_available_actions = torch.ones(
                (self.buffer_length, self.n_envs, self.n_agents, available_act_shape),
                dtype=torch.int32, device=self.device)
        elif act_space_type == 'Box':
            self.actions = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, act_shape),
                                       dtype=torch.float32, device=self.device)
            self.available_actions = torch.ones(
                (self.buffer_length, self.n_envs, self.n_agents, available_act_shape),
                dtype=torch.int32, device=self.device)
            self.next_available_actions = torch.ones(
                (self.buffer_length, self.n_envs, self.n_agents, available_act_shape),
                dtype=torch.int32, device=self.device)
        elif act_space_type == 'list':
            self.actions = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, act_shape),
                                       dtype=torch.float32, device=self.device)
            if type(available_act_shape) is tuple:
                self.available_actions = torch.ones(
                    (self.buffer_length, self.n_envs, self.n_agents, *available_act_shape),
                    dtype=torch.int32, device=self.device)
                self.next_available_actions = torch.ones(
                    (self.buffer_length, self.n_envs, self.n_agents, *available_act_shape),
                    dtype=torch.int32, device=self.device)
            else:
                self.available_actions = torch.ones(
                    (self.buffer_length, self.n_envs, self.n_agents, available_act_shape),
                    dtype=torch.int32, device=self.device)
                self.next_available_actions = torch.ones(
                    (self.buffer_length, self.n_envs, self.n_agents, available_act_shape),
                    dtype=torch.int32, device=self.device)
        else:
            raise NotImplementedError('Action space type not supported!')

        self.step = 0
        # Store priorities per environment transition (no per-agent separation)
        self.priorities = torch.zeros((self.buffer_length, self.n_envs), dtype=torch.float32, device=self.device) + 1.0
        self.max_priority = 1.0

    def __len__(self):
        return min(self.step, self.buffer_length)

    def insert(self, data):
        obs, state, next_obs, next_state, actions, rewards, masks, available_actions, next_available_actions = data
        cur_step = self.step % self.buffer_length
        self.obs[cur_step] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # Global state comes in without an agent dimension.
        self.state[cur_step] = torch.as_tensor(state.squeeze(1), dtype=torch.float32, device=self.device)
        self.next_obs[cur_step] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self.next_state[cur_step] = torch.as_tensor(next_state.squeeze(1), dtype=torch.float32, device=self.device)
        self.actions[cur_step] = torch.as_tensor(actions, device=self.device)
        if available_actions is not None:
            self.available_actions[cur_step] = torch.as_tensor(available_actions, device=self.device)
        if next_available_actions is not None:
            self.next_available_actions[cur_step] = torch.as_tensor(next_available_actions, device=self.device)
        self.rewards[cur_step - self.shift_reward] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.masks[cur_step] = torch.as_tensor(masks, dtype=torch.int64, device=self.device)
        self.priorities[cur_step].fill_(self.max_priority)
        self.step += 1

    def handle_post_data(self):
        cur_step = self.step % self.buffer_length
        self.next_obs[cur_step - 1] = self.obs[cur_step]
        self.next_state[cur_step - 1] = self.state[cur_step]

    def sample(self, alpha=0.6, beta=0.4):
        active_length = len(self)
        total_transitions = (active_length- self.shift_reward) * self.n_envs

        step = self.step % self.buffer_length
        to_be_removed = torch.arange(step - self.shift_reward, step, device=self.device)
        current_indices = torch.arange(active_length, device=self.device)
        mask = torch.ones_like(current_indices)
        mask[to_be_removed] = 0
        current_indices = current_indices[mask == 1]

        # Flatten priorities over time and environments.
        flat_priorities = self.priorities[current_indices].reshape(-1)
        probs = (flat_priorities + 1e-6) ** alpha
        probs /= probs.sum()

        sampled_indices = torch.multinomial(probs, num_samples=self.batch_size, replacement=False)
        weights = (total_transitions * probs[sampled_indices]) ** (-beta)
        weights /= weights.max()

        # Global state: shape (active_length*n_envs, state_dim)
        state = self.state[current_indices].reshape(-1, *self.state.shape[2:]).to(self.training_device)
        next_state = self.next_state[current_indices].reshape(-1, *self.next_state.shape[2:]).to(self.training_device)
        # Local observations: shape (active_length*n_envs, n_agents, obs_dim)
        obs = self.obs[current_indices].reshape(-1, self.n_agents, *self.obs.shape[3:]).to(self.training_device)
        next_obs = self.next_obs[current_indices].reshape(-1, self.n_agents, *self.next_obs.shape[3:]).to(self.training_device)
        # Actions: shape (active_length*n_envs, n_agents, act_dim)
        actions = self.actions[current_indices].reshape(-1, self.n_agents, *self.actions.shape[3:]).to(self.training_device)
        rewards = self.rewards[current_indices].reshape(-1, *self.rewards.shape[2:]).to(self.training_device)
        masks = self.masks[current_indices].reshape(-1, *self.masks.shape[2:]).to(self.training_device)
        available_actions = self.available_actions[current_indices].reshape(-1, self.n_agents, *self.available_actions.shape[3:]).to(self.training_device)
        next_available_actions = self.next_available_actions[current_indices].reshape(-1, self.n_agents, *self.next_available_actions.shape[3:]).to(self.training_device)

        # Select only the sampled transitions.
        state = state[sampled_indices]
        next_state = next_state[sampled_indices]
        obs = obs[sampled_indices]
        next_obs = next_obs[sampled_indices]
        actions = actions[sampled_indices]
        rewards = rewards[sampled_indices]
        masks = masks[sampled_indices]
        available_actions = available_actions[sampled_indices]
        next_available_actions = next_available_actions[sampled_indices]
        weights=weights.to(self.training_device)
        return obs, state, next_obs, next_state, actions, rewards, masks, available_actions, next_available_actions, sampled_indices, weights

    def update_priorities(self, indices, new_priorities):
        indices = indices.to(self.device)
        new_priorities = new_priorities.to(self.device)
        flat_priorities = self.priorities.reshape(-1)
        flat_priorities[indices] = new_priorities + 1e-6
        self.priorities = flat_priorities.reshape(self.priorities.shape)
        self.max_priority = max(self.max_priority, new_priorities.max().item())

    def get_average_rewards(self, n):
        if self.step >= self.buffer_length:
            step = self.step % self.buffer_length
            inds = torch.arange(step - n, step)
        else:
            inds = torch.arange(max(0, self.step - n), self.step)
        return self.rewards[inds].mean()
