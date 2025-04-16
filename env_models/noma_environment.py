import math
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import torch
from triton.language import dtype

from env_models.base_env import BaseEnvironment


class NomaEnvironment(BaseEnvironment):
    def __init__(self, configs):
        super(NomaEnvironment, self).__init__(configs)
        # ue_pos = self.rng.uniform(self.bs_config.bs_ue_guard, self.bs_config.coverage / 2, (self.tot_users, 2))
        # self.users_pos = torch.from_numpy(np.round(ue_pos, 2)).to(**self.tpdv)
        noise_hz = 10 ** ((self.bs_config.noise_variance - 30) / 10)
        self.noise = noise_hz * self.bs_config.bandwidth
        self.normalized_gain = torch.zeros((self.n_sbs,self.max_users, self.channel_buffer_size)).to(**self.tpdv)
        self.continuous_tx_power=False
        self.chosen_power_ratio = torch.ones((self.n_sbs,self.max_users ), dtype=torch.float32).to(self.device) # used only if we use continuous-power
        self.chosen_power_ratio[self.tot_user_mask]=0.0
        power_ratios = torch.arange(0, self.env_config.power_quant + 1) / self.env_config.power_quant
        self.power_ratios = power_ratios.repeat((self.n_sbs,self.max_users, 1)).to(**self.tpdv)
        self.cur_power_idx = torch.ones((self.n_sbs,self.max_users), dtype=torch.int64).to(self.device) * (
            self.env_config.power_quant)
        self.cur_power_idx[self.tot_user_mask]=0.0
        self.max_data_rate= self.bs_config.max_data_rate
        self.user_signals = torch.ones((self.n_sbs, self.max_users)).to(**self.tpdv)
        self.power_diff = torch.zeros((self.n_sbs, self.max_users)).to(**self.tpdv)
        self.user_sinrs = torch.zeros((self.n_sbs, self.max_users)).to(**self.tpdv)
        self.user_rates = torch.zeros((self.n_sbs, self.max_users)).to(**self.tpdv)
        self.user_rates_real = torch.zeros((self.n_sbs, self.max_users)).to(**self.tpdv)

        self.update_gain_val()
        self.perform_noma_step()

    def perform_noma_step(self):
        self.update_users_signal()
        self.update_sinr()
        self.update_users_rate()

    def prepare_next_noma_step(self):
        if (self.cur_timeslot+1) % self.channel_buffer_size == 0:
            self.update_gain_val()
        self.bg_users.bernoulli_(self.ue_config.bg_arrival_prob)
        self.bg_users[self.tot_user_mask] = 0

    @property
    def metrics_tx(self):
        ready_to_transmit = self.bg_users.int()
        # ready_to_transmit_idx=torch.argwhere(ready_to_transmit)
        num_ready=ready_to_transmit.count_nonzero()
        if num_ready:
            user_power = self.user_max_tx_power * self.chosen_power_ratio
            user_power[ready_to_transmit==0]=0
            p_watts_clamped = torch.clamp(user_power, min=1e-12)
            p_dbm = 10.0 * torch.log10(p_watts_clamped) + 30.0
            tot_power = torch.mean(p_dbm)
            mean_rate = (self.user_rates[ready_to_transmit==1]/self.max_data_rate).mean()
            sum_rate = self.user_rates[ready_to_transmit==1].sum()
            # mean_rate = self.user_rates[ready_to_transmit==1].mean()
            num_failed_pdsc = (self.power_diff < self.tolerable_power).count_nonzero()/self.tot_users
            return mean_rate.item(), num_failed_pdsc.item(),tot_power.item(),sum_rate.item()
        else:
            return None


    def update_gain_val(self):
        # ---------------------------
        # 1. Generate Random Distances (in meters)
        # ---------------------------
        min_dist = 1.0  # minimum distance in meters (avoid 0 to prevent log10(0))
        max_dist = 50.0  # maximum distance in meters
        distances = min_dist + (max_dist - min_dist) * torch.rand(
            self.n_sbs,self.max_users, self.channel_buffer_size, device=self.device
        )
        # Convert distances to kilometers for the path loss formula
        distances_km = distances / 1000.0

        # ---------------------------
        # 2. Compute Path Loss
        # ---------------------------
        # Path loss in dB: PL(dB) = 128.1 + 36.7*log10(dist[km])
        path_loss_dB = 128.1 + 36.7 * torch.log10(distances_km)
        # Convert from dB to linear scale:
        path_loss_linear = 10 ** (path_loss_dB / 10)

        # ---------------------------
        # 3. Rayleigh Fading
        # ---------------------------
        # Generate |h|^2 as an exponential random variable (unit mean)
        fading_power = -torch.log(torch.rand(self.n_sbs,self.max_users, self.channel_buffer_size, device=self.device) + 1e-10)

        # ---------------------------
        # 4. Compute Raw Gain
        # ---------------------------
        # Model: raw_gain = (fading power) / (path loss in linear scale)
        raw_gain = fading_power / path_loss_linear

        # ---------------------------
        # 5. Compute Noise Power (sigma^2)
        # ---------------------------
        # Convert noise PSD from -174 dBm/Hz to Watts/Hz:
        noise_psd_dBm_per_Hz = -174
        noise_psd_W_per_Hz = 10 ** ((noise_psd_dBm_per_Hz - 30) / 10)
        # Use 4 MHz as the system bandwidth
        effective_bandwidth = self.bs_config.bandwidth/self.num_sc * 1e6
        sigma2 = noise_psd_W_per_Hz * effective_bandwidth

        # ---------------------------
        # 6. Compute Normalized Gain
        # ---------------------------
        # Normalized gain (|h|^2 normalized by noise power)
        self.normalized_gain = raw_gain / sigma2
        self.normalized_gain[self.tot_user_mask] = 1.0


    def mean_n_gains_db(self, n):
        inds = torch.arange(self.cur_timeslot%self.channel_buffer_size - n, self.cur_timeslot%self.channel_buffer_size)
        return torch.mean(10*torch.log10(self.normalized_gain[:,:,inds]),dim=2)

    def update_users_signal(self):
        user_power = self.user_max_tx_power * self.chosen_power_ratio
        cur_second = int(self.cur_timeslot // self.slots_per_coherence)
        ready_to_transmit=self.bg_users.int()
        not_ready = ready_to_transmit==0
        users_gains = self.normalized_gain[:,:, cur_second % self.channel_buffer_size].clone()
        users_gains[not_ready] = 0
        self.user_signals = users_gains * user_power


    def update_sinr(self):
        # Get the valid (assigned) user entries per base station, subchannel and slot index.
        mask = self.user_sc_assignment != -1
        # Allocate a tensor to hold the signal values for each subchannel
        # Shape: (n_sbs, num_sc, sc_capacity)
        sc_signal = torch.zeros((self.n_sbs, self.num_sc, self.sc_capacity),
                                device=self.device, dtype=torch.float32)
        bs_indices = (torch.arange(self.n_sbs, device=self.device)
                      .unsqueeze(1).unsqueeze(2)
                      .expand(self.n_sbs, self.num_sc, self.sc_capacity))
        sc_signal[mask] = self.user_signals[bs_indices[mask], self.user_sc_assignment[mask]]

        # Sort the subchannel signals in descending order (largest first) and keep the sort indices.
        sorted_vals, sort_idx = torch.sort(sc_signal, dim=2, descending=True)

        # Prepare tensors to store the computed SINR and power difference values in the sorted order.
        sorted_sinr = torch.zeros_like(sorted_vals)
        sorted_pd = torch.zeros_like(sorted_vals)

        # We now iterate over each base station and subchannel to simulate SIC
        # (sc_capacity is usually small so the loop is acceptable).
        for i in range(self.n_sbs):
            for j in range(self.num_sc):
                failed_sum = 0.0  # Accumulated (not cancelled) interference from users that failed SIC.
                # Total interference from users that have not been attempted will be computed on the fly.
                for k in range(self.sc_capacity):
                    # Get the kth strongest signal (p_k)
                    p_val = sorted_vals[i, j, k].item()
                    # Compute the sum of signals from users not yet processed (i.e. indices > k)
                    if k < self.sc_capacity - 1:
                        S_remaining = torch.sum(sorted_vals[i, j, k + 1:]).item()
                    else:
                        S_remaining = 0.0
                    # The effective interference seen when decoding this user:
                    interference = failed_sum + S_remaining
                    # Compute the power difference (p_val - interference)
                    pdiff = p_val - interference
                    sorted_pd[i, j, k] = pdiff
                    # Compute SINR (using a noise floor of 1 as in your original code)
                    sorted_sinr[i, j, k] = p_val / (interference + 1.0)
                    # If this user fails the power disparity and sensitivity constraint,
                    # its signal is not cancelled and is added to the interference for later users.
                    if pdiff < self.tolerable_power:
                        failed_sum += p_val
                # End of loop over k (users in this subchannel)
        # End of loops over base stations and subchannels

        # Map the computed values back to the original (unsorted) order.
        # The sort_idx tensor holds, for each (i,j), the indices that would sort the original sc_signal
        # into sorted_vals along dimension 2. We can “unsort” by scattering the results.
        unsorted_sinr = torch.zeros_like(sorted_sinr)
        unsorted_pd = torch.zeros_like(sorted_pd)
        unsorted_sinr.scatter_(2, sort_idx, sorted_sinr)
        unsorted_pd.scatter_(2, sort_idx, sorted_pd)

        # Finally, assign the computed SINRs and power differences back to the corresponding users.
        # bs_indices and self.user_sc_assignment (both of shape (n_sbs, num_sc, sc_capacity)) give us the mapping.
        self.user_sinrs[bs_indices[mask], self.user_sc_assignment[mask]] = unsorted_sinr[mask]
        self.power_diff[bs_indices[mask], self.user_sc_assignment[mask]] = unsorted_pd[mask]

    def assign_users_to_subchannels(self,A):
        n_bs, n_user_per_bs = A.shape

        # One-hot encode the chosen subchannels.
        one_hot = F.one_hot(A, num_classes=self.num_sc) # shape: (n_bs, n_user_per_bs, n_subchannel)

        # Compute the cumulative sum along the user dimension.
        cumsum = torch.cumsum(one_hot, dim=1)

        # Gather the cumulative count for the chosen subchannel for each user and subtract one to get a 0-indexed rank.
        rank = cumsum.gather(2, A.unsqueeze(-1)).squeeze(-1) - 1  # shape: (n_bs, n_user_per_bs)

        # Check if any rank exceeds the allowed subchannel_capacity.
        if rank.max().item() >= self.sc_capacity:
            raise ValueError(
                "A subchannel has been assigned more users than its capacity. "
                "Please ensure that each subchannel is chosen at most `subchannel_capacity` times per base station."
            )

        # Create the output tensor B.
        B = torch.empty(n_bs, self.num_sc, self.sc_capacity, dtype=torch.long, device=A.device)

        # Create indices for base stations and user indices.
        user_idx = torch.arange(n_user_per_bs, device=A.device).unsqueeze(0).expand(n_bs, -1)
        bs_idx = torch.arange(n_bs, device=A.device).unsqueeze(1).expand(n_bs, n_user_per_bs)

        # Scatter the user indices into B using advanced indexing.
        B[bs_idx, A, rank] = user_idx

        return B

    def update_users_rate(self):
        bandwidth = self.bs_config.bandwidth/self.num_sc
        shannon_capacity = bandwidth * torch.log2(1 + self.user_sinrs)
        failed_pdsc = self.power_diff < self.tolerable_power
        shannon_capacity[failed_pdsc] = 1e-6
        shannon_capacity=torch.clamp(shannon_capacity,1e-6,max=self.max_data_rate)
        # shannon_capacity=torch.clamp(shannon_capacity,30,max=31)
        self.user_rates = torch.round(shannon_capacity, decimals=7)


