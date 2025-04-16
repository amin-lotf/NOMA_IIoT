import torch
import numpy as np
from isac.sac_utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class DQNPriorityBuffer(object):
    def __init__(self, args,n_agents, obs_space, act_space,available_act_shape, device,shift_reward=0):
        self.buffer_length = args.buffer_length
        self.batch_size = args.batch_size
        self.device = device
        self.shift_reward = shift_reward
        # Number of parallel environments and agents per environment
        self.n_envs = args.n_envs
        self.n_agents = n_agents

        # Get observation shape and create obs offloading_buffer with extra agent dim.
        obs_shape = get_shape_from_obs_space(obs_space)
        if isinstance(obs_shape[-1], list):
            obs_shape = obs_shape[:1]
        # New shape: [offloading_episode_length+1, n_envs, n_agents, *obs_shape]
        self.obs = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, *obs_shape),
                               dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, *obs_shape),
                               dtype=torch.float32, device=self.device)

        # Rewards: [offloading_episode_length, n_envs, n_agents, 1]
        self.rewards = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, 1),dtype=torch.float32,device=self.device)

        # Masks: [offloading_episode_length+1, n_envs, n_agents, 1]
        self.masks = torch.zeros((self.buffer_length , self.n_envs, self.n_agents, 1),
                                 dtype=torch.int64, device=self.device)
        # Action space
        act_shape = get_shape_from_act_space(act_space)
        act_space_type = act_space.__class__.__name__

        if act_space_type == 'MultiDiscrete':
            # Actions now have shape: [offloading_episode_length, n_envs, n_agents, act_shape]
            self.actions = torch.zeros((self.buffer_length, self.n_envs, self.n_agents, act_shape), dtype=torch.int64,device=self.device)
            # Available actions: [offloading_episode_length+1, n_envs, n_agents, act_shape]
            if type(available_act_shape) is tuple:
                self.available_actions = torch.ones(
                    (self.buffer_length, self.n_envs, self.n_agents, *available_act_shape),
                    dtype=torch.int32, device=self.device)
            else:
                self.available_actions = torch.ones(
                    (self.buffer_length, self.n_envs, self.n_agents, available_act_shape),
                    dtype=torch.int32, device=self.device)
        else:
            raise NotImplementedError('Action space type not supported!')

        # Initialize priorities for each transition.
        # We use one priority per [buffer_index, env, agent] transition.
        self.priorities = torch.zeros((self.buffer_length, self.n_envs, self.n_agents),
                                      dtype=torch.float32, device=self.device)
        self.max_priority = 1.0  # Initial maximum priority.
        self.step = 0


    def __len__(self):
        return min(self.step,self.buffer_length)

    def insert(self, data):
        obs, next_obs, actions, rewards, masks,available_actions = data
        cur_step= self.step % self.buffer_length
        if torch.is_tensor(obs):
            self.obs[cur_step].copy_(obs)
        else:
            self.obs[cur_step] = torch.as_tensor(obs).to(self.device).float()

        # Save the state information.
        if torch.is_tensor(next_obs):
            self.next_obs[cur_step].copy_(next_obs)
        else:
            self.next_obs[cur_step] = torch.as_tensor(next_obs).to(self.device).float()

        if torch.is_tensor(actions):
            self.actions[cur_step].copy_(actions)
        else:
            self.actions[cur_step] = torch.as_tensor(actions).to(self.device)

        if available_actions is not None:
            if torch.is_tensor(available_actions):
                self.available_actions[cur_step].copy_(available_actions)
            else:
                self.available_actions[cur_step].copy_(torch.as_tensor(available_actions).to(self.device))

        if torch.is_tensor(rewards):
            self.rewards[cur_step-self.shift_reward].copy_(rewards)
        else:
            self.rewards[cur_step-self.shift_reward].copy_(torch.as_tensor(rewards).to(self.device))

        self.priorities[cur_step].fill_(self.max_priority)
        self.step+=1

    def handle_post_data(self):
        self.next_obs[self.step-1 ].copy_(self.obs[self.step])

    def sample(self, alpha=0.6, beta=0.4):
        active_length = len(self)
        total_transitions = active_length * self.n_envs * self.n_agents # total transitions



        # You must make sure the buffer is bigger than shift_reward, otherwise this line will
        # cause error
        cur_step = self.step % self.buffer_length
        to_be_removed=torch.arange(cur_step-self.shift_reward,cur_step,device=self.device)
        current_indices=torch.arange(active_length,device=self.device)
        mask=torch.ones_like(current_indices)
        mask[to_be_removed]=1
        current_indices=current_indices[mask==1]

        # Flatten priorities and compute sampling probabilities.
        flat_priorities = self.priorities[current_indices].reshape(-1)
        probs = (flat_priorities + 1e-6) ** alpha
        probs /= probs.sum()

        # Sample indices based on the computed probability distribution.
        indices = torch.multinomial(probs, num_samples=self.batch_size, replacement=False)

        # Compute importance sampling weights.
        weights = (total_transitions * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        obs = self.obs.reshape(-1, *self.obs.shape[3:])
        next_obs = self.next_obs.reshape(-1, *self.next_obs.shape[3:])
        actions = self.actions.reshape(-1, *self.actions.shape[3:])
        rewards = self.rewards.reshape(-1, *self.rewards.shape[3:])
        masks = self.masks.reshape(-1, *self.masks.shape[3:])
        available_actions = self.available_actions.reshape(-1, *self.available_actions.shape[3:])
        obs = obs[indices]
        next_obs = next_obs[indices]
        actions = actions[indices]
        rewards = rewards[indices]
        masks = masks[indices]
        available_actions = available_actions[indices]
        return obs, next_obs, actions, rewards, masks,available_actions,indices,weights

    def update_priorities(self, indices, new_priorities):
        # indices is a flat array; update the corresponding positions in self.priorities
        indices = indices.to(self.device)
        new_priorities = new_priorities.to(self.device)
        flat_priorities = self.priorities.reshape(-1)
        flat_priorities[indices] = new_priorities.to(self.device) + 1e-6  # add epsilon to avoid zero
        self.priorities = flat_priorities.reshape(self.priorities.shape)
        self.max_priority = max(self.max_priority, new_priorities.max().item())


    def get_average_rewards(self, n):
        if self.step >= self.buffer_length:
            step=self.step%self.buffer_length
            inds = torch.arange(step - n, step)  # allow for negative indexing
        else:
            inds = torch.arange(max(0, self.step - n), self.step)
        return self.rewards[inds].mean()
