import torch
import numpy as np

from ppo.ppo_utils.util import get_shape_from_obs_space, get_shape_from_act_space
# If you have a dedicated get_shape_from_state_space, you could import that here.

def _flatten(T, N, A, x):
    """
    Flatten the first three dimensions [T, N, A, ...] -> [T*N*A, ...].
    """
    return x.reshape(T * N * A, *x.shape[3:])

def _cast(x):
    """
    Used in some original code for older re-shaping or transposing.
    Keep if needed, or remove if not used in new logic.
    """
    return x.transpose(1, 0).reshape(-1, *x.shape[1:])

class SharedReplayBuffer(object):
    def __init__(self, args,n_agents, obs_space, state_space, act_space,available_act_shape, device,shift_reward=0):
        self.episode_length = args.episode_length+shift_reward
        # shift reward is used for task offloading and it's equal to the task deadline,
        # we use it to make sure the EP can be calculated after all tasks for a specific slot
        # are either processed or dropped
        self.shift_reward=shift_reward
        self.hidden_size = args.hidden_size
        self.recurrent_n = args.recurrent_n
        self.device = device
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_valuenorm = args.use_valuenorm

        # Number of parallel environments and agents per environment
        self.n_envs = args.n_envs
        self.n_agents = n_agents

        # Get observation shape and create obs offloading_buffer with extra agent dim.
        obs_shape = get_shape_from_obs_space(obs_space)
        if isinstance(obs_shape[-1], list):
            obs_shape = obs_shape[:1]
        # New shape: [offloading_episode_length+1, n_envs, n_agents, *obs_shape]
        self.obs = torch.zeros((self.episode_length + 1, self.n_envs, self.n_agents, *obs_shape),
                               dtype=torch.float32, device=self.device)

        # Get state shape and create state offloading_buffer for centralized/agent state.
        state_shape = get_shape_from_obs_space(state_space)  # Or use get_shape_from_state_space()
        if isinstance(state_shape[-1], list):
            state_shape = state_shape[:1]
        # New shape: [offloading_episode_length+1, n_envs, n_agents, *state_shape]
        self.state = torch.zeros((self.episode_length + 1, self.n_envs, self.n_agents, *state_shape),
                                 dtype=torch.float32, device=self.device)

        # RNN states now include agent dimension:
        # [offloading_episode_length+1, n_envs, n_agents, recurrent_n, hidden_size]
        self.rnn_states = torch.zeros((self.episode_length + 1, self.n_envs, self.n_agents,
                                         self.recurrent_n, self.hidden_size),
                                         dtype=torch.float32, device=self.device)
        self.rnn_states_critic = torch.zeros_like(self.rnn_states)

        # Value predictions and returns per agent: shape [offloading_episode_length+1, n_envs, n_agents, 1]
        self.value_preds = torch.zeros((self.episode_length + 1, self.n_envs, self.n_agents, 1),
                                       dtype=torch.float32, device=self.device)
        self.returns = torch.zeros_like(self.value_preds)

        # Action space
        act_shape = get_shape_from_act_space(act_space)
        act_space_type = act_space.__class__.__name__

        if act_space_type == 'Discrete' or act_space_type == 'MultiDiscrete':
            # Actions now have shape: [offloading_episode_length, n_envs, n_agents, act_shape]
            self.actions = torch.zeros((self.episode_length, self.n_envs, self.n_agents, act_shape), dtype=torch.int64,device=self.device)
            # Available actions: [offloading_episode_length+1, n_envs, n_agents, act_shape]
            self.available_actions = torch.ones((self.episode_length + 1, self.n_envs, self.n_agents, available_act_shape),
                                                dtype=torch.int32, device=self.device)
            # Log-probs: [offloading_episode_length, n_envs, n_agents, 1]
            self.action_log_probs = torch.zeros((self.episode_length, self.n_envs, self.n_agents, 1),
                                                dtype=torch.float32, device=self.device)
        elif act_space_type == 'Box':
            self.actions = torch.zeros((self.episode_length, self.n_envs, self.n_agents, act_shape), dtype=torch.int64,device=self.device)
            self.available_actions = torch.ones((self.episode_length + 1, self.n_envs, self.n_agents, available_act_shape),
                                                dtype=torch.int32, device=self.device)
            self.action_log_probs = torch.zeros((self.episode_length, self.n_envs, self.n_agents, 1),
                                                dtype=torch.float32, device=self.device)
        elif act_space_type == 'list':
            self.actions = torch.zeros((self.episode_length, self.n_envs, self.n_agents, act_shape), dtype=torch.int64,device=self.device)
            self.available_actions = torch.ones((self.episode_length + 1, self.n_envs, self.n_agents, *available_act_shape),
                                                dtype=torch.int32, device=self.device)
            self.action_log_probs = torch.zeros((self.episode_length, self.n_envs, self.n_agents, 1),
                                                dtype=torch.float32, device=self.device)
        else:
            raise NotImplementedError('Action space type not supported!')

        # Rewards: [offloading_episode_length, n_envs, n_agents, 1]
        self.rewards = torch.zeros((self.episode_length, self.n_envs, self.n_agents, 1), dtype=torch.float32,device=self.device)

        # Masks: [offloading_episode_length+1, n_envs, n_agents, 1]
        self.masks = torch.zeros((self.episode_length + 1, self.n_envs, self.n_agents, 1),
                                 dtype=torch.int64, device=self.device)

        self.step = 0

    def insert(self,
               obs,
               state,
               rnn_states_actor,
               rnn_states_critic,
               actions,
               action_log_probs,
               value_preds,
               rewards,
               masks,
               available_actions=None):
        """
        Inserts one timestep of data for all agents across all environments.
        Expects each argument to have a batch dimension of [n_envs, n_agents, ...].
        """
        # Save the observation for all agents.
        if torch.is_tensor(obs):
            self.obs[self.step + 1]=obs.clone()
        else:
            self.obs[self.step + 1] = torch.as_tensor(obs).to(self.device).float()

        # Save the state information.
        if torch.is_tensor(state):
            self.state[self.step + 1]=state.clone()
        else:
            self.state[self.step + 1] = torch.as_tensor(state).to(self.device).float()

        # Save the RNN states (assumed per–agent).
        self.rnn_states[self.step + 1]=rnn_states_actor.clone()
        self.rnn_states_critic[self.step + 1].copy_(rnn_states_critic)

        # Actions might be provided as a NumPy array or tensor.
        if torch.is_tensor(actions):
            self.actions[self.step] = actions.clone()
        else:
            self.actions[self.step] = torch.as_tensor(actions).to(self.device)

        if available_actions is not None:
            if torch.is_tensor(available_actions):
                self.available_actions[self.step + 1]=available_actions.clone()
            else:
                self.available_actions[self.step + 1]=torch.as_tensor(available_actions).to(self.device)

        self.action_log_probs[self.step]=action_log_probs.clone()
        self.value_preds[self.step]=value_preds

        if torch.is_tensor(rewards):
            self.rewards[self.step-self.shift_reward] = rewards.clone()
        else:
            self.rewards[self.step-self.shift_reward] = torch.as_tensor(rewards).to(self.device)

        self.masks[self.step + 1]=torch.as_tensor(masks).to(self.device)

        # Increase step (cyclic offloading_buffer)
        self.step = (self.step + 1) % self.episode_length

    def insert_post_data(self,obs,state,available_actions=None):
        # We use this function if agents, e.g. Tx agent, needs to update their observations again  before taking
        # a step l
        if torch.is_tensor(obs):
            self.obs[self.step ]=obs
        else:
            self.obs[self.step] = torch.as_tensor(obs).to(self.device).float()

        # Save the state information.
        if torch.is_tensor(state):
            self.state[self.step ]=state
        else:
            self.state[self.step] = torch.as_tensor(state).to(self.device).float()
        if available_actions is not None:
            if torch.is_tensor(available_actions):
                self.available_actions[self.step + 1]=available_actions
            else:
                self.available_actions[self.step + 1]=torch.as_tensor(available_actions).to(self.device)





    def after_update(self):
        """
        After the update, move the last step's data to index 0 for each environment and agent.
        """
        self.obs[0].copy_(self.obs[-1])
        self.state[0].copy_(self.state[-1])
        self.rnn_states[0].copy_(self.rnn_states[-1])
        self.rnn_states_critic[0].copy_(self.rnn_states_critic[-1])
        self.masks[0].copy_(self.masks[-1])
        self.available_actions[0].copy_(self.available_actions[-1])
        self.value_preds[0].copy_(self.value_preds[-1])
        # Optionally, add other buffers if needed

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns using either vectorized GAE or simple returns.
        Assumes next_value has shape [n_envs, n_agents, 1].
        """
        self.value_preds[-1] = next_value
        rewards = self.rewards  # [T, n_envs, n_agents, 1]
        masks = self.masks                              # [T+1, n_envs, n_agents, 1]
        value_preds = self.value_preds                  # [T+1, n_envs, n_agents, 1]

        # Initialize GAE per environment and per agent.
        gae = torch.zeros((self.n_envs, self.n_agents, 1), dtype=torch.float32, device=self.device)
        for step in reversed(range(self.episode_length)):
            if self._use_valuenorm and value_normalizer is not None:
                delta = (rewards[step] +
                         self.gamma * value_normalizer.denormalize(value_preds[step + 1]) *
                         masks[step + 1] -
                         value_normalizer.denormalize(value_preds[step]))
                gae = delta + self.gamma * self.gae_lambda * masks[step + 1] * gae
                self.returns[step] = gae + value_normalizer.denormalize(value_preds[step])
            else:
                delta = (rewards[step] +
                         self.gamma * value_preds[step + 1] * masks[step + 1] -
                         value_preds[step])
                gae = delta + self.gamma * self.gae_lambda * masks[step + 1] * gae
                self.returns[step] = gae + value_preds[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield mini-batches of flattened data for feed-forward (MLP) policies.
        The data is flattened from [T, n_envs, n_agents, ...] -> [T*n_envs*n_agents, ...].
        Now also returns the flattened state.
        """
        T = self.episode_length-self.shift_reward
        N = self.n_envs
        A = self.n_agents
        batch_size = T * N * A  # total transitions

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires that the number of mini-batches is smaller than or equal to the batch size"
            )
            mini_batch_size = batch_size // num_mini_batch

        # Flatten the tensors. Note: we discard the last timestep for obs, state, etc.
        shift_adjuster= -self.shift_reward if self.shift_reward >0 else T
        obs = self.obs[:-1-self.shift_reward].reshape(T * N * A, *self.obs.shape[3:])
        state = self.state[:-1-self.shift_reward].reshape(T * N * A, *self.state.shape[3:])
        rnn_states = self.rnn_states[:-1-self.shift_reward].reshape(T * N * A, *self.rnn_states.shape[4:])
        rnn_states_critic = self.rnn_states_critic[:-1-self.shift_reward].reshape(T * N * A, *self.rnn_states_critic.shape[4:])
        actions = self.actions[:shift_adjuster].reshape(T * N * A, *self.actions.shape[3:])
        available_actions = self.available_actions[:-1-self.shift_reward].reshape(T * N * A, *self.available_actions.shape[3:])
        action_log_probs = self.action_log_probs[:shift_adjuster].reshape(T * N * A, *self.action_log_probs.shape[3:])
        value_preds = self.value_preds[:-1-self.shift_reward].reshape(T * N * A, *self.value_preds.shape[3:])
        returns = self.returns[:-1-self.shift_reward].reshape(T * N * A, *self.returns.shape[3:])
        masks = self.masks[:-1-self.shift_reward].reshape(T * N * A, *self.masks.shape[3:])
        if advantages is not None:
            advantages = advantages[:shift_adjuster].reshape(T * N * A, 1)

        # Create random mini-batch indices.
        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            indices = torch.LongTensor(indices).to(self.device)
            obs_batch = obs[indices]
            state_batch = state[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            available_actions_batch = available_actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            # Yielding state right after obs.
            yield (obs_batch,
                   state_batch,
                   rnn_states_batch,
                   rnn_states_critic_batch,
                   actions_batch,
                   value_preds_batch,
                   return_batch,
                   masks_batch,
                   old_action_log_probs_batch,
                   adv_targ,
                   available_actions_batch)

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """
        Yield mini-batches of sequential data for RNN-based policies.
        Data is broken into chunks of length `data_chunk_length` from the flattened [T, n_envs, n_agents, ...] data.
        Now also returns the corresponding state for each mini-batch.
        """
        T = self.episode_length-self.shift_reward
        N = self.n_envs
        A = self.n_agents
        batch_size = T * N * A
        # Number of chunks across all steps/agents.
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        shift_adjuster= -self.shift_reward if self.shift_reward >0 else T
        obs = self.obs[:-1-self.shift_reward].reshape(T * N * A, *self.obs.shape[3:])
        state = self.state[:-1-self.shift_reward].reshape(T * N * A, *self.state.shape[3:])
        rnn_states = self.rnn_states[:-1-self.shift_reward].reshape(T * N * A, self.recurrent_n, self.hidden_size)
        rnn_states_critic = self.rnn_states_critic[:-1-self.shift_reward].reshape(T * N * A, self.recurrent_n, self.hidden_size)
        actions = self.actions[:shift_adjuster].reshape(T * N * A, *self.actions.shape[3:])
        available_actions = self.available_actions[:-1-self.shift_reward].reshape(T * N * A, *self.available_actions.shape[3:])
        action_log_probs = self.action_log_probs[:shift_adjuster].reshape(T * N * A, *self.action_log_probs.shape[3:])
        value_preds = self.value_preds[:-1-self.shift_reward].reshape(T * N * A, *self.value_preds.shape[3:])
        returns = self.returns[:-1-self.shift_reward].reshape(T * N * A, *self.returns.shape[3:])
        masks = self.masks[:-1-self.shift_reward].reshape(T * N * A, *self.masks.shape[3:])
        if advantages is not None:
            advantages = advantages[:shift_adjuster].reshape(T * N * A, 1)

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            obs_batch_list = []
            state_batch_list = []
            rnn_states_batch_list = []
            rnn_states_critic_batch_list = []
            actions_batch_list = []

            value_preds_batch_list = []
            return_batch_list = []
            masks_batch_list = []
            old_action_log_probs_batch_list = []
            adv_batch_list = []
            available_actions_batch_list = []  # If needed

            for index in indices:
                start = index * data_chunk_length
                end = start + data_chunk_length

                obs_batch_list.append(obs[start:end])
                state_batch_list.append(state[start:end])
                actions_batch_list.append(actions[start:end])
                available_actions_batch_list.append(available_actions[start:end])
                value_preds_batch_list.append(value_preds[start:end])
                return_batch_list.append(returns[start:end])
                masks_batch_list.append(masks[start:end])
                old_action_log_probs_batch_list.append(action_log_probs[start:end])
                if advantages is not None:
                    adv_batch_list.append(advantages[start:end])
                # For RNN states, take the initial state of the chunk.
                rnn_states_batch_list.append(rnn_states[start])
                rnn_states_critic_batch_list.append(rnn_states_critic[start])

            # Stack along a new mini-batch dimension.
            obs_batch = torch.stack(obs_batch_list, dim=1)
            state_batch = torch.stack(state_batch_list, dim=1)
            actions_batch = torch.stack(actions_batch_list, dim=1)
            available_actions_batch = torch.stack(available_actions_batch_list, dim=1)
            value_preds_batch = torch.stack(value_preds_batch_list, dim=1)
            return_batch = torch.stack(return_batch_list, dim=1)
            masks_batch = torch.stack(masks_batch_list, dim=1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch_list, dim=1)
            if advantages is not None:
                adv_batch = torch.stack(adv_batch_list, dim=1)
            else:
                adv_batch = None

            rnn_states_batch = torch.stack(rnn_states_batch_list, dim=0)
            rnn_states_critic_batch = torch.stack(rnn_states_critic_batch_list, dim=0)

            L, M = data_chunk_length, mini_batch_size

            obs_batch = obs_batch.reshape(L * M, *obs_batch.shape[2:])
            state_batch = state_batch.reshape(L * M, *state_batch.shape[2:])
            actions_batch = actions_batch.reshape(L * M, *actions_batch.shape[2:])
            available_actions_batch = available_actions_batch.reshape(L * M, *available_actions_batch.shape[2:])
            value_preds_batch = value_preds_batch.reshape(L * M, *value_preds_batch.shape[2:])
            return_batch = return_batch.reshape(L * M, *return_batch.shape[2:])
            masks_batch = masks_batch.reshape(L * M, *masks_batch.shape[2:])
            old_action_log_probs_batch = old_action_log_probs_batch.reshape(L * M, *old_action_log_probs_batch.shape[2:])
            if adv_batch is not None:
                adv_batch = adv_batch.reshape(L * M, 1)
            else:
                adv_batch = None

            # available_actions_batch = None  # Update if needed

            yield (obs_batch,
                   state_batch,
                   rnn_states_batch,
                   rnn_states_critic_batch,
                   actions_batch,
                   value_preds_batch,
                   return_batch,
                   masks_batch,
                   old_action_log_probs_batch,
                   adv_batch,
                   available_actions_batch)
