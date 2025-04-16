import torch
import torch.nn.functional as F
import numpy as np
from isac.sac_algorithms.utlis.util import compute_target_entropy


class SACTrainer:
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.policy = policy
        self.gamma = args.gamma
        self.tau = args.tau  # target network smoothing coefficient
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.max_grad_norm = args.max_grad_norm if hasattr(args, 'max_grad_norm') else None

        if isinstance(policy.act_space, list):
            self.target_entropy = compute_target_entropy(policy.act_space)
        else:
            self.target_entropy = -np.prod(policy.act_space.shape)

        self.log_alpha = torch.tensor(np.log(args.alpha_init), dtype=torch.float32,
                                      requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.lr_alpha)
        self.alpha = self.log_alpha.exp().detach()

    def project_distribution(self, next_distr, rewards, masks, gamma, support, delta_z, Vmin, Vmax):
        batch_size = rewards.size(0)
        num_atoms = support.size(0)
        Tz = rewards + gamma * masks * support.view(1, -1)
        Tz = Tz.clamp(Vmin, Vmax)
        b = (Tz - Vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        l = l.clamp(0, num_atoms - 1)
        u = u.clamp(0, num_atoms - 1)
        proj_distr = torch.zeros_like(next_distr)
        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size, device=rewards.device).long().unsqueeze(1)
        next_distr = next_distr.detach()
        l_index = (l + offset).view(-1)
        u_index = (u + offset).view(-1)
        b = b.view(-1)
        next_distr = next_distr.view(-1)
        proj_distr = proj_distr.view(-1)
        proj_distr.index_add_(0, l_index, next_distr * (u.float().view(-1) - b))
        proj_distr.index_add_(0, u_index, next_distr * (b - l.float().view(-1)))
        proj_distr = proj_distr.view(batch_size, num_atoms)
        return proj_distr

    def sac_update(self, sample):
        (obs_batch, state_batch, next_obs_batch, next_state_batch, actions_batch, rewards_batch, masks_batch,
         available_actions_batch, next_available_actions_batch, indices, weights) = sample

        # === Critic update (centralized) ===
        # First, compute next actions from the next local observations.
        # Here next_obs_batch has shape (batch, n_agents, obs_dim) so we flatten it to process each agent separately.
        batch_size, n_agents, obs_dim = next_obs_batch.shape
        next_obs_flat = next_obs_batch.reshape(batch_size * n_agents, obs_dim)
        next_available_actions_flat = next_available_actions_batch.reshape(batch_size * n_agents, *next_available_actions_batch.shape[2:])
        next_actions_flat, next_log_pi_flat, _ = self.policy.actor(next_obs_flat, next_available_actions_flat, deterministic=False)
        next_actions = next_actions_flat.reshape(batch_size, n_agents, -1)
        next_log_pi = next_log_pi_flat.reshape(batch_size, n_agents, -1)
        # Form the joint action by concatenating actions from all agents.
        joint_next_actions = next_actions.reshape(batch_size, -1)

        with torch.no_grad():
            target_distr1, target_distr2 = self.policy.critic_target(next_state_batch, joint_next_actions)
            target_distr = (target_distr1 + target_distr2) / 2.0
            Vmin = self.policy.critic.Vmin
            Vmax = self.policy.critic.Vmax
            delta_z = self.policy.critic.delta_z
            support = self.policy.critic.support
            projected_distr = self.project_distribution(target_distr, rewards_batch, masks_batch, self.gamma, support,
                                                        delta_z, Vmin, Vmax)

        # For the current critic update, flatten the sampled per-agent actions into a joint action.
        # actions_batch: (batch, n_agents, act_dim)
        joint_actions = actions_batch.reshape(batch_size, -1)
        current_distr1, current_distr2 = self.policy.critic(state_batch, joint_actions)
        critic_loss_1 = F.kl_div(current_distr1.log(), projected_distr, reduction='batchmean')
        critic_loss_2 = F.kl_div(current_distr2.log(), projected_distr, reduction='batchmean')
        critic_loss = (critic_loss_1 + critic_loss_2)
        critic_loss = (critic_loss * weights).mean()

        self.policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.policy.critic_optimizer.step()

        # === Actor update (decentralized) ===
        # Process local observations: flatten so that each agent is processed individually.
        batch_size, n_agents, obs_dim = obs_batch.shape
        obs_flat = obs_batch.reshape(batch_size * n_agents, obs_dim)
        available_actions_flat = available_actions_batch.reshape(batch_size * n_agents, *available_actions_batch.shape[2:])
        actions_new_flat, log_pi_flat, _ = self.policy.actor(obs_flat, available_actions_flat, deterministic=False)
        actions_new = actions_new_flat.reshape(batch_size, n_agents, -1)
        log_pi = log_pi_flat.reshape(batch_size, n_agents, -1)
        # Form joint actions for the critic by concatenating.
        joint_actions_new = actions_new.reshape(batch_size, -1)
        q1_new_dist, q2_new_dist = self.policy.critic(state_batch, joint_actions_new)
        q1_new = self.policy.critic.get_expectation(q1_new_dist)
        q2_new = self.policy.critic.get_expectation(q2_new_dist)
        q_new = torch.min(q1_new, q2_new)
        # Sum the log probabilities over agents to get the joint log probability.
        joint_log_pi = log_pi.sum(dim=1)
        actor_loss = (self.alpha * joint_log_pi - q_new).mean()

        self.policy.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.policy.actor_optimizer.step()

        # === Temperature (alpha) update ===
        alpha_loss = -(self.log_alpha * (joint_log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # === Soft update of critic target ===
        for param, target_param in zip(self.policy.critic.parameters(), self.policy.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        with torch.no_grad():
            current_q1 = self.policy.critic.get_expectation(current_distr1)
            target_q = torch.sum(projected_distr * support, dim=1, keepdim=True)
            td_error = torch.abs(current_q1 - target_q).squeeze(1)

        return critic_loss.item(), actor_loss.item(), alpha_loss.item(), self.alpha.item(), indices, td_error

    def train(self, replay_buffer):
        train_info = {"critic_loss": 0, "actor_loss": 0, "alpha_loss": 0, "alpha": 0}
        for _ in range(self.epochs):
            sample = replay_buffer.sample()
            critic_loss, actor_loss, alpha_loss, alpha_val, indices, td_error = self.sac_update(sample)
            train_info["critic_loss"] += critic_loss
            train_info["actor_loss"] += actor_loss
            train_info["alpha_loss"] += alpha_loss
            train_info["alpha"] = alpha_val
            replay_buffer.update_priorities(indices, td_error)
        for k in train_info:
            train_info[k] /= self.epochs
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
