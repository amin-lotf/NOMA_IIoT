import torch
import numpy as np
from isac.sac_utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class PrioritizedSACBuffer(object):
    def __init__(self, args,n_agents, obs_space, state_space, act_space,available_act_shape, device,shift_reward=0):
        self.buffer_length = args.buffer_length
        self.batch_size = args.batch_size
        # self.device = torch.device('cpu')
        self.device = device
        self.training_device = device
        self.shift_reward = shift_reward
        # Number of parallel environments and agents per environment
        self.n_envs = args.n_envs
        self.n_agents = n_agents
        # Get observation shape and create obs offloading_buffer with extra agent dim.
        obs_shape = get_shape_from_obs_space(obs_space)
        if isinstance(obs_shape[-1], list):
            obs_shape = obs_shape[:1]

        state_shape = get_shape_from_obs_space(state_space)
        if isinstance(state_shape[-1], list):
            state_shape = state_shape[:1]
        # New shape: [offloading_episode_length+1, n_envs, n_agents, *obs_shape]
        self.obs = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, *obs_shape),
                               dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, *obs_shape),
                                    dtype=torch.float32, device=self.device)

        self.state = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, *state_shape),
                               dtype=torch.float32, device=self.device)

        self.next_state = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, *state_shape),
                                    dtype=torch.float32, device=self.device)

        # Rewards: [offloading_episode_length, n_envs, n_agents, 1]
        self.rewards = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, 1), dtype=torch.float32,
                                   device=self.device)

        # Masks: [offloading_episode_length+1, n_envs, n_agents, 1]
        self.masks = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, 1),
                                 dtype=torch.int64, device=self.device)
        # Action space
        act_shape = get_shape_from_act_space(act_space)
        act_space_type = act_space.__class__.__name__

        if act_space_type == 'Discrete' or act_space_type == 'MultiDiscrete':
            # Actions now have shape: [offloading_episode_length, n_envs, n_agents, act_shape]
            self.actions = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, act_shape), dtype=torch.int64,
                                       device=self.device)
            # Available actions: [offloading_episode_length+1, n_envs, n_agents, act_shape]
            self.available_actions = torch.ones(
                (self.buffer_length, self.n_envs, self.n_agents, available_act_shape),
                dtype=torch.int32, device=self.device)
            self.next_available_actions = torch.ones(
                (self.buffer_length, self.n_envs, self.n_agents, available_act_shape),
                dtype=torch.int32, device=self.device)
        elif act_space_type == 'Box':
            self.actions = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, act_shape), dtype=torch.float32,
                                       device=self.device)
            self.available_actions = torch.ones(
                (self.buffer_length, self.n_envs, self.n_agents, available_act_shape),
                dtype=torch.int32, device=self.device)
            self.next_available_actions = torch.ones(
                (self.buffer_length, self.n_envs, self.n_agents, available_act_shape),
                dtype=torch.int32, device=self.device)
        elif act_space_type == 'list':
            self.actions = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, act_shape), dtype=torch.float32,
                                       device=self.device)
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
        # Initialize priorities for each transition; start with max priority (e.g., 1)
        self.priorities = torch.zeros((self.buffer_length, self.n_envs, self.n_agents), dtype=torch.float32,device=self.device) + 1.0

    def __len__(self):
        return min(self.step, self.buffer_length)

    def insert(self, data):
        obs,state, next_obs,next_state, actions, rewards, masks, available_actions, next_available_actions = data
        cur_step = self.step % self.buffer_length
        self.obs[cur_step] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.state[cur_step] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.next_obs[cur_step] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self.next_state[cur_step] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.actions[cur_step] = torch.as_tensor(actions, device=self.device)
        if available_actions is not None:
            self.available_actions[cur_step] = torch.as_tensor(available_actions, device=self.device)
        if next_available_actions is not None:
            self.next_available_actions[cur_step] = torch.as_tensor(next_available_actions, device=self.device)
        self.rewards[cur_step-self.shift_reward] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.masks[cur_step] = torch.as_tensor(masks, dtype=torch.int64, device=self.device)
        # For new transitions, set max priority
        self.priorities[cur_step] = 1.0
        self.step += 1

    def handle_post_data(self):
        cur_step = self.step % self.buffer_length
        self.next_obs[cur_step - 1 ] = self.obs[cur_step]
        self.next_state[cur_step - 1 ] = self.state[cur_step]

    def sample(self):
        T = self.batch_size
        N = self.n_envs
        A = self.n_agents
        tot_batch_size= T*N*A
        # tot_batch_size= T
        total_transitions = len(self)
        # if T * N * A > total_transitions:
        #     raise ValueError('Batch size must be smaller than current buffer length')

        # Exclude the latest `shift_reward` transitions from sampling
        active_length = total_transitions
        to_be_removed = torch.arange(self.step - self.shift_reward, self.step, device=self.device)
        current_indices = torch.arange(active_length, device=self.device)
        mask = torch.ones_like(current_indices, dtype=torch.bool)
        mask[to_be_removed] = False  # Set `False` for indices to be removed
        valid_indices = current_indices[mask]  # Keep only valid transitions

        obs = self.obs[valid_indices].reshape(-1, *self.obs.shape[3:])
        next_obs = self.next_obs[valid_indices].reshape(-1, *self.next_obs.shape[3:])
        state = self.state[valid_indices].reshape(-1, *self.state.shape[3:])
        next_state = self.next_state[valid_indices].reshape(-1, *self.next_state.shape[3:])
        actions = self.actions[valid_indices].reshape(-1, *self.actions.shape[3:])
        rewards = self.rewards[valid_indices].reshape(-1, *self.rewards.shape[3:])
        masks = self.masks[valid_indices].reshape(-1, *self.masks.shape[3:])
        available_actions = self.available_actions[valid_indices].reshape(-1, *self.available_actions.shape[3:])
        next_available_actions = self.next_available_actions[valid_indices].reshape(-1, *self.next_available_actions.shape[3:])

        # Flatten the priorities for sampling
        flat_priorities = self.priorities[valid_indices].reshape(-1)
        probs = flat_priorities + 1e-6  # Ensure no zero probability
        probs /= probs.sum()

        # sampled_indices1 = np.random.choice(valid_indices.cpu().numpy(), size=T * N * A, p=probs.cpu().numpy())
        sampled_relative_indices = torch.multinomial(probs, tot_batch_size, replacement=False)
        valid_indices_expanded = torch.arange(flat_priorities.shape[0], device=self.device)
        # Map back to the original valid indices
        sampled_indices = valid_indices_expanded[sampled_relative_indices]

        # Fetch the sampled transitions
        obs = obs[sampled_indices].to(self.training_device)
        next_obs = next_obs[sampled_indices].to(self.training_device)
        state = state[sampled_indices].to(self.training_device)
        next_state = next_state[sampled_indices].to(self.training_device)
        actions = actions[sampled_indices].to(self.training_device)
        rewards = rewards[sampled_indices].to(self.training_device)
        masks = masks[sampled_indices].to(self.training_device)
        available_actions = available_actions[sampled_indices].to(self.training_device)
        next_available_actions = next_available_actions[sampled_indices].to(self.training_device)

        # Compute importance-sampling weights
        weights = 1.0 / (flat_priorities.shape[0] * probs[sampled_indices]).to(self.training_device)
        # weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        return obs,state, next_obs,next_state, actions, rewards, masks, available_actions,next_available_actions, sampled_indices, weights

    def update_priorities(self, indices, new_priorities):
        # indices is a flat array; update the corresponding positions in self.priorities
        indices=indices.to(self.device)
        new_priorities=new_priorities.to(self.device)
        flat_priorities = self.priorities.reshape(-1)
        flat_priorities[indices] = new_priorities.to(self.device) + 1e-6  # add epsilon to avoid zero
        self.priorities = flat_priorities.reshape(self.priorities.shape)

    def get_average_rewards(self, n):
        if self.step >= self.buffer_length:
            step = self.step % self.buffer_length
            inds = torch.arange(step - n, step)
        else:
            inds = torch.arange(max(0, self.step - n), self.step)
        return self.rewards[inds].mean()
