import torch
import torch.nn as nn

from duel_ddqn.duel_ddqn_algorithms.r_madqn.algorithm.utlis.distributions import LinearDuelingDQN, LinearDQN

class MultiDiscreteDQNActLayer(nn.Module):
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, device, metadata: dict = None, args=None):
        super(MultiDiscreteDQNActLayer, self).__init__()
        self.device = device

        # choose the network type
        dqn = LinearDuelingDQN if args.use_dueling else LinearDQN

        # Create one DQN head for each discrete action dimension.
        self.action_dims = action_space.high - action_space.low + 1
        self.action_outs = nn.ModuleList([
            dqn(inputs_dim, int(action_dim), use_orthogonal, gain)
            for action_dim in self.action_dims
        ])

        # Handle metadata for special action types
        if metadata is not None:
            act_type = metadata.get('act_type', False)
            self.is_single_agent = metadata.get('is_single_agent', False)
            self.num_agents = metadata.get('num_agents', False)
            self.n_sbs = metadata.get('n_sbs', False)
            self.max_users = metadata['max_users']
            if act_type == 'noma':
                self.is_noma = True
                self.is_offloading = False
                self.n_sc = metadata['n_sc']
                self.sc_capacity = metadata['sc_capacity']
            elif act_type == 'offloading':
                self.is_noma = False
                self.is_offloading = True
                self.num_splits = metadata['split_quantization']
            else:
                self.is_noma = False
                self.is_offloading = False
        else:
            self.is_noma = False
            self.is_offloading = False

    def forward(self, x, epsilon, available_actions=None):
        """
        Forward pass that selects actions using epsilon-greedy.
        """
        actions = []
        q_values_all = []  # storing branch Q-values if needed

        # This helper function replaces the per-sample loop.
        # It selects a random valid action uniformly if epsilon is met,
        # and falls back to the greedy action if no valid actions exist.
        def _select_final_action(q_values, epsilon):
            # q_values: [B, n_actions]
            # Compute the greedy (max-Q) action:
            greedy = q_values.argmax(dim=1, keepdim=True)
            # Determine which actions are valid (assumed valid if q > -1e8)
            valid = q_values > -1e8

            # Generate random scores for each action
            r = torch.rand_like(q_values)
            # Mask invalid actions by setting them to -infty so they are never chosen:
            r = r.masked_fill(~valid, float('-inf'))
            # Pick the index with the highest random score in each row
            random_act = r.argmax(dim=1, keepdim=True)
            # For rows with no valid actions, use the greedy action
            no_valid = valid.sum(dim=1, keepdim=True) == 0
            random_act = torch.where(no_valid, greedy, random_act)

            # Now choose between random and greedy using epsilon-greedy
            rand_sample = torch.rand(q_values.size(0), 1, device=q_values.device)
            choose_random = (rand_sample < epsilon)
            final = torch.where(choose_random, random_act, greedy)
            return final

        if self.is_noma:
            # For NOMA, we update a subchannel allocation statistic.
            sc_stat = torch.zeros((x.shape[0], self.n_sc), device=self.device)
            for idx, action_out in enumerate(self.action_outs):
                if idx < self.max_users:
                    # --- Subchannel Allocation branch ---
                    act_mask = (sc_stat < self.sc_capacity)
                    q_values = action_out(x, act_mask)
                    if available_actions is not None:
                        # For subchannel allocation, assume mask is at available_actions[:, 0, idx:idx+1]
                        avail_act = available_actions[:,  idx:idx + 1]
                        q_values = q_values * avail_act + (1 - avail_act) * (-1e9)
                    # Vectorized random selection:
                    final_action = _select_final_action(q_values, epsilon)
                    if available_actions is not None:
                        avail_act = available_actions[:, idx:idx + 1]
                        final_action = final_action * avail_act + (1 - avail_act) * -1
                        q_values = q_values * avail_act + (1 - avail_act) * (-1e9)
                    actions.append(final_action)
                    q_values_all.append(q_values)
                    # Update sc_stat: for each sample where final_action != -1, increment the count.
                    row_indices = torch.arange(sc_stat.shape[0], device=self.device).unsqueeze(-1)
                    mask = final_action != -1
                    sc_stat[row_indices[mask], final_action[mask]] += 1
                else:
                    # --- Power Allocation branch (or another branch) ---
                    q_values = action_out(x)
                    if available_actions is not None:
                        # For power allocation assume the mask comes in channel 2.
                        avail_act = available_actions[:,  (idx - self.max_users):(idx - self.max_users) + 1]
                        q_values = q_values * avail_act + (1 - avail_act) * (-1e9)
                    final_action = _select_final_action(q_values, epsilon)
                    if available_actions is not None:
                        avail_act = available_actions[:,  (idx - self.max_users):(idx - self.max_users) + 1]
                        q_values = final_action * avail_act + (1 - avail_act) * (-1)
                    actions.append(final_action)
                    q_values_all.append(q_values)

        elif self.is_offloading:
            # For offloading, use a split mask based on the remaining splits.
            ue_count = self.n_sbs * self.max_users // self.num_agents
            split_stat = torch.arange(self.num_splits + 1, device=self.device) \
                .reshape(1, 1, -1).expand(x.shape[0], ue_count, -1)
            taken_splits = torch.zeros((x.shape[0], ue_count, 1), device=self.device)
            for idx, action_out in enumerate(self.action_outs):
                if idx < self.max_users:
                    ue_idx = idx
                    rem_splits = self.num_splits - taken_splits[:, ue_idx, :]
                    act_mask = split_stat[:, ue_idx, :] <= rem_splits
                    q_values = action_out(x, act_mask)
                    if available_actions is not None:
                        avail_act = available_actions[:, idx:idx + 1]
                        q_values = q_values * avail_act + (1 - avail_act) * (-1e9)
                    final_action = _select_final_action(q_values, epsilon)
                    if available_actions is not None:
                        avail_act = available_actions[:, idx:idx + 1]
                        final_action = final_action * avail_act
                    actions.append(final_action)
                    q_values_all.append(q_values)
                    taken_splits[:, ue_idx, :] += final_action.float()
                else:
                    # --- MEC Power Allocation branch (or another branch) ---
                    q_values = action_out(x)
                    if available_actions is not None:
                        avail_act = available_actions[:, (idx - self.max_users):(idx - self.max_users) + 1]
                        q_values = q_values * avail_act + (1 - avail_act) * (-1e9)
                    final_action = _select_final_action(q_values, epsilon)
                    if available_actions is not None:
                        avail_act = available_actions[:, (idx - self.max_users):(idx - self.max_users) + 1]
                        q_values = final_action * avail_act + (1 - avail_act) * (-1)
                    actions.append(final_action)
                    q_values_all.append(q_values)

        else:
            # Standard case: no special masking beyond available_actions.
            for idx, action_out in enumerate(self.action_outs):
                act_mask = None
                q_values = action_out(x, act_mask)
                if available_actions is not None:
                    avail_act = available_actions[:, idx:idx + 1]
                    q_values = q_values * avail_act + (1 - avail_act) * (-1e9)
                final_action = _select_final_action(q_values, epsilon)
                actions.append(final_action)
                q_values_all.append(q_values)

        # Concatenate the selected actions along the last dimension.
        actions_cat = torch.cat(actions, dim=-1)
        # Return a dummy second output (None) to preserve a PPO-like interface.
        return actions_cat

    def evaluate_actions(self, x, action, available_actions=None):
        action_vals = []
        for idx, action_out in enumerate(self.action_outs):
            if self.is_noma or self.is_offloading:
                act = action[:, idx].reshape((-1, 1))
                if available_actions is not None:
                    act_idx = idx % self.max_users
                    act = act * available_actions[:,  act_idx:act_idx + 1]
                action_val = action_out(x).gather(1, act)
                if available_actions is not None:
                    action_val = action_val * available_actions[:, act_idx:act_idx + 1]
            else:
                act = action[:, idx].reshape((-1, 1))
                if available_actions is not None:
                    act = act * available_actions[:, idx:idx + 1]
                action_val = action_out(x).gather(1, act)
                if available_actions is not None:
                    action_val = action_val * available_actions[:, idx:idx + 1]
            action_vals.append(action_val)
        action_vals = torch.cat(action_vals, -1)
        return action_vals
