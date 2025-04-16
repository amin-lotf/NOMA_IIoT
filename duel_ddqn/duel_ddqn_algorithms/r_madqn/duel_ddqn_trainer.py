import numpy as np
import torch
from duel_ddqn.duel_ddqn_utils.valuenorm import ValueNorm
# torch.autograd.set_detect_anomaly(True)
class MADQNTrainer:
    def __init__(self, args, eval_policy,target_policy, device=torch.device("cpu")):
        self.device = device
        self.args=args
        # self.use_dueling=args.use_dueling
        self.eval_policy = eval_policy
        self.target_policy=target_policy
        self._use_valuenorm = args.use_valuenorm
        self.learn_step_counter=0
        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
        self.test_init=True
        self.q_val=None
        self.q_next=None

    def train(self, buffer, update_actor=True, alpha=0.6, beta=0.4):
        train_info = {}
        train_info['value_loss'] = 0
        # ... (other logging)

        # Periodically update target network.
        if self.learn_step_counter % self.args.q_network_iter == 0:
            self.target_policy.actor.load_state_dict(self.eval_policy.actor.state_dict())
        self.learn_step_counter += 1

        # Sample from PER buffer (which returns indices and IS weights).
        data = buffer.sample(alpha=alpha, beta=beta)
        obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch, available_actions_batch, indices, weights = data

        # Get Q-values; these are of shape (batch_size, num_actions)
        q_eval = self.eval_policy.evaluate_actions(obs_batch, actions_batch, available_actions_batch)

        # Calculate target Q-values
        with torch.no_grad():
            next_actions = self.eval_policy.get_batch_acts(next_obs_batch, available_actions_batch)
            q_next = self.target_policy.evaluate_actions(next_obs_batch, next_actions, available_actions_batch)
            q_next[(dones_batch == 1).flatten(), :] = 0
            q_target = rewards_batch + self.args.dqn_gamma * q_next

        # Compute element-wise TD errors
        td_errors = torch.abs(q_target - q_eval).detach()  # Shape: (batch_size, num_actions)

        # Aggregate TD errors per transition (e.g., mean over num_actions)
        aggregated_td_error = td_errors.mean(dim=1)  # Shape: (batch_size)

        # Compute the MSE loss for each element and then weight by IS weights.
        # Option 1: Compute loss per element and then average (if flattening)
        # loss = self.eval_policy.loss_func(q_eval.flatten(), q_target.flatten())

        # Option 2: Compute loss per sample by aggregating over actions:
        sample_loss = ((q_eval - q_target) ** 2).mean(dim=1)  # Loss per sample, shape: (batch_size)
        weighted_loss = (sample_loss * weights).mean()

        # Backpropagate weighted loss.
        self.eval_policy.actor_optimizer.zero_grad()
        weighted_loss.backward()
        self.eval_policy.actor_optimizer.step()

        # Update PER priorities using aggregated TD errors.
        new_priorities = aggregated_td_error
        buffer.update_priorities(indices, new_priorities)

        train_info['value_loss'] = weighted_loss.item()
        return train_info

    def prep_training(self,train_all=True):
        self.eval_policy.actor.train()
        # if not train_all:
        #     # self.eval_policy.actor.base.mlp_hidden.eval()
        #     self.eval_policy.actor.base.mlp_hidden.eval()


    def prep_rollout(self):
        self.eval_policy.actor.eval()
