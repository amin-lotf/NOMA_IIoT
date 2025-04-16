import numpy as np
import torch
from gymnasium import spaces

from env_models.mec_environment import MECEnvironment



class PSOEnv(MECEnvironment):
    def __init__(self, configs):
        super(PSOEnv, self).__init__(configs)
        self.pgoa_offloading_config = configs['pso_offloading_config']
        self.pgoa_tx_config = configs['pso_config']
        self.observation_space=spaces.Box(low=-np.inf, high=+np.inf, shape=(0,), dtype=np.float32)
        self.action_space=spaces.Box(low=-np.inf, high=+np.inf, shape=(0,), dtype=np.float32)


    def reset(self, seed=None, options=None,step_type=0):
        super().reset(seed,options)
        info ={}
        return info


    def step(self, step_type=0):
        if step_type == 0:
            return self.tx_step()
        elif step_type==1:
            return self.offloading_step()
        else:
            raise NotImplementedError('actin not supported')


    def tx_step(self):
        self.optimize_resource_allocation()
        self.perform_noma_step()
        noma_metrics = self.metrics_tx
        self.prepare_next_noma_step()
        info = {'noma_metrics': noma_metrics}
        reward = self.tx_reward.cpu().numpy()
        # self.cur_timeslot+=1
        return  reward,info

    def offloading_step(self):
        self.perform_early_steps()
        res=pso_optimize_offloading_all_users_gpu(self)
        split_ratio_t=torch.as_tensor(res['task_split_ratio'], dtype=self.task_split_ratio.dtype, device=self.device)
        self.task_split_ratio = split_ratio_t * (self.ue_arrival_buffer>0)
        # self.task_split_ratio=torch.as_tensor(split_actions,dtype=self.task_split_ratio.dtype,device=self.device)
        idx = self.cur_timeslot % (self.max_task_deadline + 1)
        offloading_decision_t=torch.as_tensor(res['offloading_distribution'], dtype=self.task_offloading_decision.dtype, device=self.device)
        tot_split=split_ratio_t+offloading_decision_t.sum(-1)
        if torch.any(torch.round(tot_split,decimals=3)< 1.0):
            split_n=tot_split.cpu().numpy()
            print('error')
        self.task_offloading_decision[:,:,:,idx]=offloading_decision_t*(self.ue_arrival_buffer.unsqueeze(-1)>0)
        self.task_mec_power_ratio[:,:,:,idx]=torch.as_tensor(res['task_mec_power_ratio'], dtype=self.task_mec_power_ratio.dtype, device=self.device)
        self.perform_later_steps()
        offloading_metrics = self.metrics_offloading
        info = {'offloading_metrics': offloading_metrics}
        reward = self.offloading_reward.cpu().numpy()
        return reward, info

    def optimize_resource_allocation(self, iterations=5, target_sinr=40.0, alpha=0.1):
        """
        Iteratively updates the transmit power allocation and subchannel assignment.
        This simplified algorithm is inspired by the paper's decomposition method.

        Args:
            iterations (int): Number of optimization iterations.
            target_sinr (float): A target SINR value for adjusting power ratios.
            alpha (float): A scaling factor for the power update step.
        """
        for it in range(iterations):
            # --- Step 1: Update transmit power allocation ---
            # We use a simple heuristic: if the current SINR is below target, increase power ratio;
            # if above, decrease it. (In practice, you might solve a convex optimization.)
            current_sinr = self.user_sinrs.clone()
            # Avoid division by zero and update power ratios
            new_power_ratio = torch.where(
                current_sinr > 0,
                self.chosen_power_ratio * (target_sinr / (current_sinr + 1e-6)) ** alpha,
                self.chosen_power_ratio
            )
            # Ensure power ratios remain between 0 and 1
            self.chosen_power_ratio = torch.clamp(new_power_ratio, 0.01, 1.0)

            # --- Step 2: Update user signals and SINR based on new power allocation ---
            self.update_users_signal()
            self.update_sinr()

            # --- Step 3: Update subchannel assignment (greedy heuristic) ---
            # For each base station, reassign users to subchannels based on their averaged channel gains.
            # We aim for each subchannel to have two users where the first slot is the one with the highest gain.
            for bs in range(self.n_sbs):
                # Get indices of users that are active (bg_users == 1)
                valid_users = torch.nonzero(self.bg_users[bs]).squeeze()
                if valid_users.numel() == 0:
                    continue
                # Compute the average gain (in dB) over the channel buffer for each user
                avg_gains = self.mean_n_gains_db(n=self.channel_buffer_size)[bs]
                # Restrict to valid users
                gains_valid = avg_gains[valid_users]
                # Sort valid users in descending order of average gain
                sorted_order = torch.argsort(gains_valid, descending=True)
                sorted_user_ids = valid_users[sorted_order]
                # Create a new assignment tensor of shape (num_sc, sc_capacity)
                new_assignment = -torch.ones((self.num_sc, self.sc_capacity), dtype=torch.int64, device=self.device)
                # Only assign as many users as possible (up to num_sc*sc_capacity)
                num_to_assign = min(sorted_user_ids.numel(), self.num_sc * self.sc_capacity)
                assigned_users = sorted_user_ids[:num_to_assign]
                # Reshape into (num_sc, sc_capacity) and assign
                new_assignment = assigned_users.reshape(self.num_sc, self.sc_capacity)
                self.user_sc_assignment[bs] = new_assignment

            # --- Step 4: Recompute user signals and SINR after assignment update ---
            self.update_users_signal()
            self.update_sinr()

            # Optionally, you can print or log the intermediate energy/SINR values for monitoring.
            # For example:
            # print(f"Iteration {it+1}: Mean SINR = {self.user_sinrs.mean().item():.2f}")

        # End of iterative optimization
        return


def pso_optimize_offloading_all_users_gpu(env, iterations=500, pop_size=50):
    """
    PSO optimizer rewritten to use vectorized PyTorch operations on CUDA.
    This version uses a surrogate objective that aims to maximize an estimate of the
    offloading_reward. It computes delay and energy terms, estimates a “processed ratio”
    (i.e. successful processing), and then defines a surrogate reward similar to:
      reward = - (overall_delay) + norm_succeed_bits - mean_power - norm_wasted_bits.
    Since PSO minimizes the fitness, we return -reward.
    """
    device = env.device
    n_bs = env.n_sbs
    users_per_bs = int(env.max_users)
    total_users = n_bs * users_per_bs
    d_user = 1 + 2 * n_bs  # per-user decision: first 1+n_bs for split, next n_bs for MEC power ratios
    D = total_users * d_user

    # PSO hyperparameters
    w = 0.7  # inertia weight
    c1 = 1.5  # cognitive coefficient
    c2 = 1.5  # social coefficient

    # Initialize population and velocity on the GPU
    population = torch.rand(pop_size, D, device=device)
    velocity = torch.rand(pop_size, D, device=device) * 0.1

    # Global constants from env (converted to CPU scalars or tensors as needed)
    task_power = env.ue_config.task_power
    cpu_coeff = env.bs_config.cpu_coefficient
    tx_power = env.user_max_tx_power.item()
    mec_cp = env.mec_computing_power.flatten()  # shape (n_bs,)

    # Per-user parameters (moved to GPU)
    task_sizes = env.ue_arrival_buffer.to(device).reshape(-1)  # shape (total_users,)
    local_cps = env.ue_computing_power.to(device).reshape(-1)
    user_rates = env.user_rates.to(device).reshape(-1)

    eps = 1e-6

    # --- Helper functions (vectorized) ---
    def quantize_and_normalize_splits(splits):
        """
        Quantize each element to the nearest 0.25 and normalize so that the sum is 1.
        Input shape: (..., 1+n_bs)
        """
        splits = torch.clamp(splits, min=0)
        splits = torch.round(splits / 0.25) * 0.25
        ssum = splits.sum(dim=-1, keepdim=True)
        zero_mask = (ssum == 0)
        splits = torch.where(zero_mask, torch.ones_like(splits) / splits.size(-1), splits / (ssum + eps))
        return splits

    def clip_mec_ratios(ratios):
        """Clip MEC power ratios to be within [0.01, 1.0]."""
        return torch.clamp(ratios, 0.01, 1.0)

    def surrogate_fitness(pop):
        """
        Compute the surrogate fitness for a batch of candidate solutions.
        The input `pop` is of shape (pop_size, D) and is reshaped to
        (pop_size, total_users, d_user). All calculations are vectorized.
        This surrogate estimates a reward based on:
          - overall delay (which we want to be low),
          - an estimate of processed (succeed) bits vs. wasted bits,
          - and the energy (power) consumption.
        We then define fitness = - (surrogate_reward) so that lower fitness is better.
        """
        pop = pop.view(pop_size, total_users, d_user)
        # First (1+n_bs) elements: splits (local split and offloading distribution)
        splits = pop[:, :, :1 + n_bs]  # shape: (pop_size, total_users, 1+n_bs)
        # Next n_bs elements: candidate MEC power ratios
        mec_candidates = pop[:, :, 1 + n_bs:]  # shape: (pop_size, total_users, n_bs)
        splits = quantize_and_normalize_splits(splits)
        local_splits = splits[..., 0]  # local processing fraction, shape: (pop_size, total_users)
        offload_splits = splits[..., 1:]  # offloading fractions, shape: (pop_size, total_users, n_bs)
        mec_ratios = clip_mec_ratios(mec_candidates)  # shape: (pop_size, total_users, n_bs)

        # Expand per-user parameters for pop_size candidates
        ts = task_sizes.unsqueeze(0).expand(pop_size, -1)  # (pop_size, total_users)
        lc = local_cps.unsqueeze(0).expand(pop_size, -1)
        ur = user_rates.unsqueeze(0).expand(pop_size, -1)

        # Compute delays:
        local_delay = (ts * task_power) / (lc + eps)  # processing delay if done locally
        tx_delay = ts / (ur + eps)  # transmission delay
        # For offloading: delay includes transmission + remote processing delay.
        mec_cp_exp = mec_cp.unsqueeze(0).unsqueeze(0)  # shape: (1,1,n_bs)
        mec_delay = (ts.unsqueeze(-1) * task_power) / (mec_cp_exp * mec_ratios + eps)
        offload_delays = tx_delay.unsqueeze(-1) + mec_delay
        # Overall delay as a weighted sum between local and offloaded delays.
        overall_delay = local_splits * local_delay + torch.sum(offload_splits * offload_delays, dim=-1)

        # Compute energy consumption:
        local_energy = local_delay * (task_power * cpu_coeff * (lc ** 2))
        tx_energy = tx_delay * tx_power
        mec_energy = (ts.unsqueeze(-1) * task_power) / (mec_cp_exp * mec_ratios + eps) * (
                task_power * cpu_coeff * (mec_cp_exp ** 2))
        offload_energies = tx_energy.unsqueeze(-1) + mec_energy
        overall_energy = local_splits * local_energy + torch.sum(offload_splits * offload_energies, dim=-1)

        # --- Incorporate a simple model for “processing success” ---
        # Assume a maximum delay threshold (e.g., maximum allowable delay)
        max_delay = env.max_task_deadline * env.slot_duration
        # Estimate the processed ratio: if delay is low, more bits are processed.
        processed_ratio = torch.clamp(1 - overall_delay / max_delay, 0, 1)
        # Estimate succeeded and wasted bits (per user) from the arriving task size.
        succeed_bits_est = ts * processed_ratio
        wasted_bits_est = ts * (1 - processed_ratio)
        max_task_size = env.ue_config.max_task_size

        # Also estimate a “mean power” term by normalizing overall energy.
        mean_power_est = overall_energy / env.max_power_consumption

        # Define a surrogate reward (similar in spirit to your offloading_reward):
        #   reward = - (overall_delay) + normalized succeed bits - mean power - normalized wasted bits.
        # (Here we take the mean over users for each candidate solution.)
        surrogate_reward = (
                - overall_delay.mean(dim=1)
                + (succeed_bits_est / max_task_size).mean(dim=1)
                - mean_power_est.mean(dim=1)
                - (wasted_bits_est / max_task_size).mean(dim=1)
        )
        # Since PSO minimizes fitness, we set fitness = - surrogate_reward.
        fitness = -surrogate_reward
        return fitness

    # --- PSO main loop ---
    fitness_vals = surrogate_fitness(population)
    pbest = population.clone()
    pbest_fitness = fitness_vals.clone()
    best_idx = torch.argmin(fitness_vals)
    gbest = population[best_idx].clone()
    gbest_fitness = pbest_fitness[best_idx].item()

    for it in range(iterations):
        r1 = torch.rand(pop_size, D, device=device)
        r2 = torch.rand(pop_size, D, device=device)
        velocity = w * velocity + c1 * r1 * (pbest - population) + c2 * r2 * (gbest - population)
        population = torch.clamp(population + velocity, 0, 1)
        fitness_vals = surrogate_fitness(population)

        # Update personal bests and global best
        improved = fitness_vals < pbest_fitness
        pbest[improved] = population[improved].clone()
        pbest_fitness[improved] = fitness_vals[improved]
        best_idx = torch.argmin(pbest_fitness)
        if pbest_fitness[best_idx] < gbest_fitness:
            gbest = pbest[best_idx].clone()
            gbest_fitness = pbest_fitness[best_idx].item()
        # Optionally print intermediate fitness:
        # print(f"Iteration {it+1}: best fitness = {gbest_fitness:.3f}")

    # Reshape the best candidate into decisions for each user.
    best_candidate = gbest.view(total_users, d_user)
    splits_all = quantize_and_normalize_splits(best_candidate[:, :1 + n_bs])
    local_splits = splits_all[:, 0].view(n_bs, users_per_bs)
    offloading_dist = splits_all[:, 1:].view(n_bs, users_per_bs, n_bs)
    mec_all = clip_mec_ratios(best_candidate[:, 1 + n_bs:]).view(n_bs, users_per_bs, n_bs)

    result = {
        "task_split_ratio": local_splits.cpu().numpy(),
        "offloading_distribution": offloading_dist.cpu().numpy(),
        "task_mec_power_ratio": mec_all.cpu().numpy(),
        "fitness": gbest_fitness
    }
    return result








