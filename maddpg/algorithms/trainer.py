import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorizedAttentionMADDPGTrainer:
    def __init__(self, args, policy, device=torch.device("cpu"), n_agents=None):
        self.device = device
        self.policy = policy
        self.gamma = args.gamma
        self.tau = args.tau
        self.max_grad_norm = args.max_grad_norm if hasattr(args, 'max_grad_norm') else None
        self.n_agents = n_agents  # number of agents

    def maddpg_update(self, sample):
        # Sample shapes:
        # joint_obs: (B, N, obs_dim)
        # joint_next_obs: (B, N, obs_dim)
        # joint_actions: (B, N, act_dim)
        # rewards: (B, N)
        # masks: (B, N)
        # joint_available_actions, joint_next_available_actions: same per-agent shapes
        joint_obs, joint_next_obs,global_state,global_next_state, joint_actions, rewards, masks, joint_available_actions, joint_next_available_actions, sample_indices = sample
        B, N, _ = joint_obs.shape

        # Form the global state by concatenating the observations of all agents.
        # global_state = joint_obs.reshape(B, -1)         # shape: (B, N * obs_dim)
        # global_next_state = joint_next_obs.reshape(B, -1)   # shape: (B, N * obs_dim)
        # Flatten joint actions into one vector per sample.
        joint_actions_flat = joint_actions.reshape(B, -1)  # shape: (B, N * act_dim)

        # Compute target actions for next timestep:
        obs_next_flat = joint_next_obs.view(B * N, -1)
        next_available_actions_flat = joint_next_available_actions.reshape(B * N, *joint_next_available_actions.shape[2:])
        target_actions_flat = self.policy.target_actor(obs_next_flat, next_available_actions_flat, deterministic=True)
        # Flatten the per–agent target actions:
        target_actions = target_actions_flat.view(B, -1)  # shape: (B, N * act_dim)

        with torch.no_grad():
            # Here rewards and masks could be averaged (or summed) over agents if needed.
            # For simplicity we take the mean reward over agents.
            target_q = self.policy.target_critic(global_next_state, target_actions)
            y = rewards.mean(dim=1, keepdim=True) + self.gamma * target_q * masks.mean(dim=1, keepdim=True)

        # Critic update: use the current joint actions (flattened) and global state.
        current_q = self.policy.critic(global_state, joint_actions_flat)
        td_errors = torch.abs(current_q.detach() - y)
        critic_loss = F.mse_loss(current_q, y)
        self.policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.policy.critic_optimizer.step()

        # Actor update: generate current actions from the actor, flatten, and then evaluate.
        obs_flat = joint_obs.view(B * N, -1)
        available_actions_flat = joint_available_actions.reshape(B * N, *joint_available_actions.shape[2:])
        current_actions_flat = self.policy.actor(obs_flat, available_actions_flat, deterministic=True)
        current_actions = current_actions_flat.view(B, -1)
        actor_loss = -self.policy.critic(global_state, current_actions).mean()
        self.policy.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.policy.actor_optimizer.step()

        # Soft update target networks.
        self.soft_update(self.policy.critic, self.policy.target_critic)
        self.soft_update(self.policy.actor, self.policy.target_actor)

        update_info = {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}
        return update_info, td_errors, sample_indices

    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train(self, buffer):
        sample = buffer.sample()  # Expected to return a tuple of tensors.
        update_info, td_errors, sample_indices = self.maddpg_update(sample)
        new_priorities = td_errors.mean(dim=1)  # averaging over batch dimension as an example
        buffer.update_priorities(sample_indices, new_priorities)
        return update_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

