import numpy as np
import torch
import torch.nn as nn

from ppo.ppo_utils.util import get_gard_norm, mse_loss
from ppo.ppo_utils.valuenorm import ValueNorm


class PPOTrainer:
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.fedprox_mu=args.fedprox_mu
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.global_actor_par=None
        self.global_critic_par=None
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.initial_entropy_coef = args.entropy_coef  # store the initial value
        # Hyperparameters: adjust decay_rate and min_entropy_coef as needed.
        self.decay_rate = args.entropy_decay_rate  # Decay rate per episode (tune this based on your total episodes)
        self.min_entropy_coef = args.min_entropy_coef  # Lower bound to ensure some level of exploration remains
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss

        self._use_valuenorm = args.use_valuenorm


        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)
        else:
            self.value_normalizer = None

    def update_entropy_coefficient(self, episode):
        """
        Dynamically update the entropy coefficient using an exponential decay schedule.
        This helps the agent explore more at the start and gradually focus on exploitation.

        Args:
            episode (int): The current episode number.
        """


        # Exponential decay schedule with a floor.
        new_entropy_coef = self.initial_entropy_coef * np.exp(-self.decay_rate * episode)
        self.entropy_coef = max(new_entropy_coef, self.min_entropy_coef)

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        value_loss_clipped = mse_loss(error_clipped)
        value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
        value_loss = value_loss.mean()
        return value_loss

    def ppo_update(self, sample, update_actor=True):
        obs_batch,state_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample


        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
                                                                              obs_batch,
                                                                              state_batch,
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch
                                                                              )
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ


        policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        # actor_fedprox=fedprox(self.offloading_policy.actor.hidden_layer,self.global_actor_par,self.fedprox_mu) if self.global_actor_par else 0
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)

        self.policy.critic_optimizer.zero_grad()
        # critic_fedprox = fedprox(self.offloading_policy.critic.hidden_layer, self.global_critic_par,self.fedprox_mu) if self.global_critic_par else 0
        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):

        if self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.clone()
        # advantages_copy[offloading_buffer.active_masks[:-1] == 0.0] = float('nan')
        mean_advantages = torch.nanmean(advantages_copy)
        std_advantages = torch.std(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0


        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        train_info['entropy_coef'] = self.entropy_coef
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()


    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()


