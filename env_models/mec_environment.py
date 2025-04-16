import torch
from env_models.noma_environment import NomaEnvironment


def round_tensor(x, decimals=0):
    """Helper: rounds tensor x to a given number of decimals."""
    factor = 10 ** decimals
    return torch.round(x * factor) / factor


class MECEnvironment(NomaEnvironment):
    def __init__(self, configs):
        super(MECEnvironment, self).__init__(configs)
        self.max_task_deadline = self.ue_config.task_deadline
        self.sbs_range = torch.arange(self.n_sbs, device=self.device)
        self.max_users_range = torch.arange(self.max_users, device=self.device)
        self.global_users_range = torch.arange(self.global_users, device=self.device)
        # =========== Start To Be Decided Variables =========== #
        self.task_split_ratio = torch.ones((self.n_sbs, self.max_users), device=self.device)
        self.task_split_data = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1), device=self.device)
        self.task_offloading_decision = torch.zeros(
            (self.n_sbs, self.max_users, self.n_sbs, self.max_task_deadline + 1), device=self.device)
        self.task_mec_power_ratio = torch.zeros(
            (self.n_sbs, self.max_users, self.n_sbs, self.max_task_deadline + 1), device=self.device)
        # =========== End To Be Decided Variables =========== #
        self.max_task_deadline_expanded = torch.ones((self.n_sbs, self.max_users, 1),
                                                     device=self.device) * self.max_task_deadline
        self.task_power_expanded = torch.ones((self.n_sbs, self.max_users, 1),
                                              device=self.device) * self.ue_config.task_power
        self.ue_arrival_buffer = torch.zeros((self.n_sbs, self.max_users), device=self.device)
        self.task_type_buffer = torch.zeros((self.n_sbs, self.max_users), device=self.device)
        self.task_tot_size = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1), device=self.device)
        self.task_loc_size = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1), device=self.device)
        self.task_tx_size = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1), device=self.device)
        self.task_mec_size = torch.zeros((self.n_sbs, self.global_users, self.max_task_deadline + 1),
                                         device=self.device)
        self.task_deadline = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1), device=self.device)
        self.task_mec_deadline = torch.zeros((self.n_sbs, self.global_users, self.max_task_deadline + 1),
                                             device=self.device)
        self.task_type = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1), device=self.device)
        self.cur_loc_idx = torch.zeros((self.n_sbs, self.max_users, 1), device=self.device, dtype=torch.int64)
        self.cur_tx_idx = torch.zeros((self.n_sbs, self.max_users, 1), device=self.device, dtype=torch.int64)
        self.cur_mec_idx = torch.ones((self.n_sbs, self.global_users, 1), device=self.device,
                                      dtype=torch.int64) * self.max_task_deadline
        self.task_local_bits_done = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                                device=self.device)
        self.task_mec_bits_done = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                              device=self.device)
        self.task_tx_bits_done = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                             device=self.device)

        self.local_computing_power_consumed = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                                          device=self.device)
        self.local_tx_power_consumed = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                                   device=self.device)
        self.mec_computing_power_consumed = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                                        device=self.device)
        # =========== Start Stats===========#
        self.task_arrival_stat = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                             device=self.device)
        self.task_split_stat = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                           device=self.device)
        self.task_bits_arrived_stat = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                                  device=self.device)
        self.task_bits_success_stat = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                                  device=self.device)
        self.task_bits_wasted_stat = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                                 device=self.device)
        self.task_drop_done_stat = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                               device=self.device)
        self.task_delay_stat = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                           device=self.device)
        self.task_delay_sensitive_stat = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                                     device=self.device)
        self.task_delay_no_sensitive_stat = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                                        device=self.device)
        self.task_power_stat = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1),
                                           device=self.device)
        self.unprocessed_dropped_bits_stat = torch.zeros((self.max_task_deadline + 1),
                                                         device=self.device)
        self.processed_dropped_bits_stat = torch.zeros((self.max_task_deadline + 1),
                                                       device=self.device)
        # =========== End Stats===========#
        # =========== Start Min-Max Vals===========#
        self.max_ue_computing_power = self.ue_computing_power.max().item()
        self.max_mec_computing_power = self.mec_computing_power.max().item() * self.bs_config.max_allocated_power_fraction
        loc_proc_time = self.ue_config.max_task_size * self.ue_config.task_power / self.max_ue_computing_power
        max_power_watt = self.ue_config.task_power * self.ue_config.cpu_coefficient * self.max_ue_computing_power ** 2
        max_loc_joule = loc_proc_time * max_power_watt  # in Joules
        mec_proc_time = self.ue_config.max_task_size * self.ue_config.task_power / self.max_mec_computing_power
        max_power_watt = self.ue_config.task_power * self.bs_config.cpu_coefficient * self.max_mec_computing_power ** 2
        max_mec_joule = mec_proc_time * max_power_watt# in Joules
        max_tx_joule = self.user_max_tx_power * self.slot_duration
        # max_joule = max_loc_joule + max_mec_joule + max_tx_joule
        max_joule =  max_mec_joule + max_tx_joule
        self.max_power_consumption = max_joule
        # =========== End Min-Max Vals===========#
        self.done_tx_tasks = torch.zeros((self.n_sbs, self.max_users), device=self.device)
        self.mec_arrival_buffer = torch.zeros((self.n_sbs, self.max_users), device=self.device)
        # A counter for MEC offloading (per BS):
        self.mec_offloading_counter = torch.zeros((self.n_sbs,), device=self.device, dtype=torch.float32)
        # A small value to add to stats to avoid zero values for better presentation
        self.small_val = torch.tensor(1e-5, device=self.device)

    def perform_early_steps(self):
        self.record_stat()
        self.update_task_deadlines()

    def perform_later_steps(self):
        self.perform_task_split_decision()
        self.perform_local_processing()
        self.perform_mec_processing()
        self.perform_tx_processing()
        self.perform_task_offloading_decision()
        self.cur_timeslot += 1
        self.perform_task_arrival_decision()


    @property
    def energy_productivity(self):
        sensitivity_factor = self.env_config.delay_sensitivity_factor
        d = self.task_type / (
                1 + torch.exp(sensitivity_factor * (self.task_delay_stat - self.max_task_deadline / 2))) + (
                    1 - self.task_type)
        d[self.task_drop_done_stat == 0] = 0.0
        power = self.task_power_stat.sum(dim=(0, 1)) + self.small_val
        # nominator = (d * self.task_bits_success_stat-self.task_bits_wasted_stat)
        nominator = (d * self.task_bits_success_stat)
        ep = torch.clamp(nominator.sum(dim=(0, 1)),min=0.0 )/ power
        # ep = d
        # -------Note that we must get the ep after cur_timeslot is increased so  idx will point to the slot that
        # its deadline is already passed, and we can collect the stats -------#
        idx = self.cur_timeslot % (self.max_task_deadline + 1)
        # bit_s=self.task_bits_success_stat[:,:,idx].sum()
        # bit_w=self.task_bits_wasted_stat[:,:,idx].sum()
        return ep[idx]

    @property
    def is_done(self):
        return False

    @property
    def tx_reward(self):
        num_failed_pdsc = torch.count_nonzero(self.power_diff < self.tolerable_power)
        reward = self.env_config.diff_power_weight*(self.global_users - num_failed_pdsc) / (self.global_users + 0.00001)
        user_rate = self.user_rates.clone()
        mean_data_rate = torch.sum(user_rate / self.max_data_rate) / self.global_users
        user_power = self.user_max_tx_power * self.chosen_power_ratio
        mean_power = user_power.sum() / (self.global_users * self.user_max_tx_power.squeeze())
        reward += mean_data_rate - self.env_config.tx_power_weight * mean_power
        # if reward >= 2.0:
        #     reward *= 10
        if reward <= 0:
            reward -= 2
        return round_tensor(reward, decimals=3).reshape(1, -1)



    @property
    def offloading_reward(self):
        # return self.energy_productivity
        idx = self.cur_timeslot % (self.max_task_deadline + 1)
        sensitivity_factor = self.env_config.delay_sensitivity_factor
        d = self.task_type / (
                1 + torch.exp(sensitivity_factor * (self.task_delay_stat - self.max_task_deadline / 2))) + (
                    1 - self.task_type)
        d[self.task_drop_done_stat == 0] = 0.0


        delay=d[:,:,idx]
        done_mask=self.task_drop_done_stat[:,:,idx] == 1
        valid_delay=delay[done_mask]
        tot_delay_count=valid_delay.numel()
        delay_norm=torch.nan_to_num(valid_delay.sum()/tot_delay_count)
        # task_bits_arrived = torch.maximum(self.task_bits_arrived_stat[:, :, idx].sum(), self.small_val).item()
        succeed_bits = self.task_bits_success_stat[:, :, idx]
        waisted_bits = self.task_bits_wasted_stat[:, :, idx]
        valid_succeed_bits=succeed_bits[succeed_bits>0]
        valid_wasted_bits=waisted_bits[waisted_bits>0]
        tot_tasks=valid_succeed_bits.numel()+valid_wasted_bits.numel()
        sum_bits=valid_succeed_bits.sum()-valid_wasted_bits.sum()
        norm_bits=torch.nan_to_num(sum_bits/(tot_tasks*self.ue_config.max_task_size))
        # norm_succeed_bits = torch.maximum(succeed_bits / task_bits_arrived, self.small_val).item()
        # norm_succeed_bits = (succeed_bits / (self.max_users*self.ue_config.max_task_size)).item()
        # norm_waisted_bits = (waisted_bits / (self.max_users*self.ue_config.max_task_size)).item()
        chosen_power = self.task_power_stat[:, :, idx]
        valid_power = chosen_power[chosen_power > 0]
        mean_power = torch.maximum(torch.nan_to_num(valid_power.sum() / (tot_tasks*self.max_power_consumption)), self.small_val)
        reward = delay_norm + norm_bits-mean_power
        # -------Note that we must get the ep after cur_timeslot is increased so  idx will point to the slot that
        # its deadline is already passed, and we can collect the stats -------#
        return round_tensor(reward, decimals=3).reshape(1, -1)

    @property
    def metrics_offloading(self):
        idx = self.cur_timeslot % (self.max_task_deadline + 1)
        if torch.all(self.task_arrival_stat[:, :, idx] == 0):
            return None
        delays = self.task_delay_stat[:, :, idx]
        # delays = delays[delays > 0]
        mean_delay = torch.maximum(torch.nan_to_num(delays[delays > 0].mean()), self.small_val).item()
        sensitivity_delays = self.task_delay_sensitive_stat[:, :, idx]
        mean_sensitivity_delay = torch.maximum(torch.nan_to_num(sensitivity_delays[sensitivity_delays > 0].mean()),
                                               self.small_val).item()
        tot_power = torch.maximum((self.task_power_stat[:, :, idx] / self.max_power_consumption).sum(),
                                  self.small_val).item()
        chosen_power = self.task_power_stat[:, :, idx]
        valid_power = chosen_power[chosen_power > 0]
        mean_power = torch.maximum((valid_power / self.max_power_consumption).mean(), self.small_val).item()
        # norm_power = self.task_power_stat / self.max_power_consumption
        task_bits_success = self.task_bits_success_stat[:, :, idx].sum()
        done_bits = torch.maximum(task_bits_success, self.small_val).item()
        task_bits_arrived = torch.maximum(self.task_bits_arrived_stat[:, :, idx].sum(), self.small_val).item()
        num_drops = (self.task_drop_done_stat[:, :, idx] == -1).count_nonzero()
        tot_arrived = torch.maximum(self.task_arrival_stat[:, :, idx].sum(), self.small_val)
        drop_ratio = torch.maximum(num_drops / tot_arrived, self.small_val).item()
        split_stat=self.task_split_stat[:,:,idx]
        valid_split_stat=split_stat[self.task_arrival_stat[:, :, idx]>0]
        mean_split = torch.maximum(valid_split_stat.mean(), self.small_val).item()
        # bits_balance = task_bits_arrived - done_bits - wasted_bits
        ep = self.energy_productivity.item()
        unprocessed_dropped = torch.maximum(self.unprocessed_dropped_bits_stat[idx], self.small_val).item()
        processed_dropped = torch.maximum(self.processed_dropped_bits_stat[idx], self.small_val).item()
        real_dropped = unprocessed_dropped + processed_dropped
        bits_balance = task_bits_arrived - real_dropped - done_bits
        # norm_wasted_bits = torch.maximum(wasted_bits / task_bits_arrived, self.small_val).item()
        return ep, mean_delay, tot_power, task_bits_success.item(), bits_balance, drop_ratio, mean_sensitivity_delay, mean_power,mean_split

    def reset_slot_data(self):
        idx = self.cur_timeslot % (self.max_task_deadline + 1)
        self.local_computing_power_consumed[:, :, idx] = 0.0
        self.local_tx_power_consumed[:, :, idx] = 0.0
        self.mec_computing_power_consumed[:, :, idx] = 0.0
        self.task_local_bits_done[:, :, idx] = 0.0
        self.task_mec_bits_done[:, :, idx] = 0.0
        self.task_tx_bits_done[:, :, idx] = 0.0
        self.task_tot_size[:, :, idx] = 0.0
        self.task_arrival_stat[:, :, idx] = 0.0

    def record_stat(self):
        # We record everything even if it is duplicated, because at the end of the
        # slot we need to get the metrics and calculate the reward and some original values
        # may change when taking step
        mec_buffer_extended = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1), device=self.device)
        mec_buffer_extended[
            self.sbs_range.unsqueeze(-1), self.max_users_range, self.cur_tx_idx.squeeze()] = self.mec_arrival_buffer
        task_mec_size = self.task_mec_size.reshape((self.n_sbs, self.n_sbs, self.max_users, -1)).transpose(1,
                                                                                                           2).transpose(
            0, 2)
        cur_mec_task = task_mec_size.sum(dim=2)
        glob_task_size_stat = self.task_loc_size + self.task_tx_size + mec_buffer_extended + cur_mec_task
        glob_task_size_stat = torch.round(glob_task_size_stat, decimals=self.bs_config.decimal_round)
        drop_mask = (self.task_deadline == 1) & (glob_task_size_stat > 0)
        done_mask = (self.task_deadline >= 1) & (glob_task_size_stat <= 0)
        task_power_consumed = self.local_computing_power_consumed + self.local_tx_power_consumed + self.mec_computing_power_consumed
        self.task_power_stat = task_power_consumed
        delay_stat = 1 + (self.max_task_deadline - self.task_deadline.clone())
        delay_stat[~done_mask] = 0.0
        delay_stat[self.task_drop_done_stat == 1] = self.task_delay_stat[self.task_drop_done_stat == 1]
        self.task_delay_stat = delay_stat
        delay_sensitive_mask = self.task_type == 1
        delay_sensitive_stat = torch.zeros_like(self.task_delay_sensitive_stat)
        delay_sensitive_stat[delay_sensitive_mask] = delay_stat[delay_sensitive_mask]
        self.task_delay_sensitive_stat = delay_sensitive_stat
        delay_no_sensitive_stat = torch.zeros_like(self.task_delay_no_sensitive_stat)
        delay_no_sensitive_stat[~delay_sensitive_mask] = delay_stat[~delay_sensitive_mask]
        self.task_delay_no_sensitive_stat = delay_no_sensitive_stat
        task_status = torch.zeros_like(self.task_drop_done_stat)
        task_status[drop_mask] = -1.0
        task_status[done_mask] = 1.0
        self.task_drop_done_stat = task_status
        wasted_task_bits = torch.zeros_like(self.task_bits_wasted_stat)
        wasted_task_bits[drop_mask] = self.task_tot_size[drop_mask]
        self.task_bits_wasted_stat = wasted_task_bits
        done_task_bits = torch.zeros_like(self.task_bits_success_stat)
        done_task_bits[done_mask] = self.task_tot_size[done_mask]
        self.task_bits_success_stat = done_task_bits
        task_arrival_stat = torch.zeros_like(self.task_arrival_stat)
        task_arrival_stat[self.task_tot_size > 0] = 1.0
        self.task_arrival_stat = task_arrival_stat
        self.task_bits_arrived_stat = self.task_tot_size.clone()
        self.task_split_stat=self.task_split_data.clone()
        mec_buffer_extended = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1), device=self.device)
        mec_buffer_extended[
            self.sbs_range.unsqueeze(-1), self.max_users_range, self.cur_tx_idx.squeeze()] = self.mec_arrival_buffer
        remained_dropped_bits = self.task_loc_size[drop_mask].sum() + self.task_tx_size[drop_mask].sum() + \
                                cur_mec_task[drop_mask].sum() + mec_buffer_extended[drop_mask].sum()

        idx = (self.cur_timeslot + 1) % (self.max_task_deadline + 1)
        processed_dropped = self.task_local_bits_done[drop_mask].sum() + self.task_mec_bits_done[drop_mask].sum()
        self.unprocessed_dropped_bits_stat[idx] = torch.round(remained_dropped_bits,
                                                              decimals=self.bs_config.decimal_round)
        self.processed_dropped_bits_stat[idx] = torch.round(processed_dropped, decimals=self.bs_config.decimal_round)

    def update_task_deadlines(self):
        zero_tensor = torch.tensor(0, device=self.device, dtype=self.task_deadline.dtype)
        self.task_deadline = torch.maximum(self.task_deadline - 1, zero_tensor)
        self.task_mec_deadline = torch.maximum(self.task_mec_deadline - 1, zero_tensor)
        mec_buffer_extended = torch.zeros((self.n_sbs, self.max_users, self.max_task_deadline + 1), device=self.device)
        mec_buffer_extended[
            self.sbs_range.unsqueeze(-1), self.max_users_range, self.cur_tx_idx.squeeze()] = self.mec_arrival_buffer
        deadline_mask = self.task_deadline == 0
        deadline_mec_mask = self.task_mec_deadline == 0

        self.task_loc_size[deadline_mask] = 0.0
        self.task_tx_size[deadline_mask] = 0.0
        self.task_mec_size[deadline_mec_mask] = 0.0
        mec_buffer_extended[deadline_mask] = 0.0
        mec_buffer = torch.gather(mec_buffer_extended, 2, self.cur_tx_idx).squeeze()
        self.mec_arrival_buffer = mec_buffer

    def perform_task_arrival_decision(self):
        mask = torch.rand((self.n_sbs, self.max_users), device=self.device) > self.ue_config.task_arrival_probability
        type_mask = torch.rand((self.n_sbs, self.max_users), device=self.device) < self.ue_config.task_type_probability
        tasks = torch.rand((self.n_sbs, self.max_users), device=self.device) * (
                self.ue_config.max_task_size - self.ue_config.min_task_size) + self.ue_config.min_task_size
        tasks[mask] = 0.0
        type_mask[mask] = 0
        self.ue_arrival_buffer = tasks
        # Task type = 1 means the task is delay sensitive.
        self.task_type_buffer = type_mask.int()
        # idx = self.cur_timeslot % (self.max_task_deadline + 1)
        # self.task_arrival_stat[:, :, idx] = tasks

    def perform_task_split_decision(self):
        self.reset_slot_data()
        loc_tasks = self.ue_arrival_buffer * self.task_split_ratio
        remote_tasks = self.ue_arrival_buffer * (1 - self.task_split_ratio)
        idx = self.cur_timeslot % (self.max_task_deadline + 1)
        self.task_split_data[:, :, idx] = self.task_split_ratio
        self.task_loc_size[:, :, idx] = loc_tasks
        self.task_tx_size[:, :, idx] = remote_tasks
        self.task_tot_size[:, :, idx] = self.ue_arrival_buffer.clone()
        self.task_deadline[:, :, idx] = self.max_task_deadline * (self.ue_arrival_buffer != 0).to(
            self.task_deadline.dtype)
        task_deadline_extended = torch.repeat_interleave(self.task_deadline.unsqueeze(dim=2), self.n_sbs, dim=2)
        task_deadline_extended = task_deadline_extended.transpose(dim1=0, dim0=2).transpose(dim1=1, dim0=2).flatten(
            start_dim=1, end_dim=2)
        self.task_mec_deadline = task_deadline_extended
        self.task_type[:, :, idx] = self.task_type_buffer.clone()

    def perform_local_processing(self):
        slot_indices = torch.arange(self.max_task_deadline + 1, device=self.device).reshape(1, 1, -1)
        slot_indices = slot_indices.expand(self.n_sbs, self.max_users, self.max_task_deadline + 1)
        slot_indices = (slot_indices + self.cur_loc_idx) % (self.max_task_deadline + 1)
        slots_ordered = torch.gather(self.task_loc_size, 2, slot_indices)
        condition = slots_ordered != 0
        next_task_idx = torch.argmax(condition.to(torch.int64), dim=2, keepdim=True)
        next_task_idx = ((next_task_idx + self.cur_loc_idx) % (self.max_task_deadline + 1))
        cur_tasks = torch.gather(self.task_loc_size, 2, next_task_idx).squeeze()
        zero_tensor = torch.zeros_like(cur_tasks)
        rem_tasks = torch.maximum(cur_tasks - self.slot_duration * self.ue_computing_power / self.task_power,
                                  zero_tensor)
        processed_task = cur_tasks - rem_tasks
        processed_time = processed_task * self.ue_config.task_power / self.ue_computing_power
        power_watt = self.ue_config.task_power * self.bs_config.cpu_coefficient * self.ue_computing_power ** 2
        processed_energy = processed_time * power_watt  # in Joules
        # tot_energy=processed_energy.sum()
        self.task_local_bits_done[
            self.sbs_range.unsqueeze(-1), self.max_users_range, next_task_idx.squeeze()] += processed_task
        self.local_computing_power_consumed[
            self.sbs_range.unsqueeze(-1), self.max_users_range, next_task_idx.squeeze()] += processed_energy
        self.task_loc_size[self.sbs_range.unsqueeze(-1), self.max_users_range, next_task_idx.squeeze()] = torch.round(
            rem_tasks, decimals=self.bs_config.decimal_round)
        self.cur_loc_idx = next_task_idx

    def perform_tx_processing(self):
        slot_indices = torch.arange(self.max_task_deadline + 1, device=self.device).reshape(1, 1, -1)
        slot_indices = slot_indices.expand(self.n_sbs, self.max_users, self.max_task_deadline + 1)
        slot_indices = (slot_indices + self.cur_tx_idx) % (self.max_task_deadline + 1)
        slots_ordered = torch.gather(self.task_tx_size, 2, slot_indices)
        condition = slots_ordered != 0
        next_task_idx = torch.argmax(condition.to(torch.int64), dim=2, keepdim=True)
        next_task_idx = ((next_task_idx + self.cur_tx_idx) % (self.max_task_deadline + 1))
        cur_tasks = torch.gather(self.task_tx_size, 2, next_task_idx).squeeze()
        max_tx_capacity = torch.round(self.user_rates * self.slot_duration,decimals=4)
        zero_tensor = torch.zeros_like(cur_tasks)
        rem_tasks = torch.maximum(cur_tasks - max_tx_capacity, zero_tensor)
        transmitted_task = cur_tasks - rem_tasks
        req_time = transmitted_task / self.user_rates
        user_power = self.user_max_tx_power*self.chosen_power_ratio
        req_tx_energy = req_time * user_power
        # tot_energy = req_tx_energy.sum()
        self.task_tx_bits_done[
            self.sbs_range.unsqueeze(-1), self.max_users_range, next_task_idx.squeeze()] += transmitted_task
        self.local_tx_power_consumed[
            self.sbs_range.unsqueeze(-1), self.max_users_range, next_task_idx.squeeze()] += req_tx_energy
        self.task_tx_size[self.sbs_range.unsqueeze(-1), self.max_users_range, next_task_idx.squeeze()] = rem_tasks
        done_tx_task = rem_tasks == 0
        self.done_tx_tasks = torch.bitwise_and(done_tx_task, transmitted_task > 0)
        self.mec_arrival_buffer += transmitted_task.squeeze()
        # self.mec_arrival_buffer=self.task_tot_size[self.sbs_range.unsqueeze(-1),self.max_users_range,self.cur_tx_idx.squeeze()]
        # self.mec_arrival_buffer[~self.done_tx_tasks]=0.0
        self.cur_tx_idx = next_task_idx

    def perform_task_offloading_decision(self):
        # Determine tasks to be offloaded (only where transmission is done)
        # to_be_offloaded = self.mec_arrival_buffer * self.done_tx_tasks.float()  # shape: [n_sbs, max_users]
        to_be_offloaded = self.task_tot_size[self.sbs_range.unsqueeze(
            -1), self.max_users_range, self.cur_tx_idx.squeeze()] * self.done_tx_tasks.float()  # shape: [n_sbs, max_users]
        to_be_offloaded = torch.repeat_interleave(to_be_offloaded.unsqueeze(-1), self.n_sbs, dim=-1)
        to_be_offloaded = to_be_offloaded.transpose(0, 2).transpose(1, 2).flatten(start_dim=1, end_dim=2)
        offloading_decision = self.task_offloading_decision.transpose(0, 2).transpose(1, 2).flatten(start_dim=1,
                                                                                                    end_dim=2)
        cur_tx_idx = torch.repeat_interleave(self.cur_tx_idx.unsqueeze(2), self.n_sbs, dim=2).transpose(0, 2).transpose(
            1, 2).flatten(start_dim=1, end_dim=2).squeeze()
        offloading_decision = offloading_decision[self.sbs_range.unsqueeze(-1), self.global_users_range, cur_tx_idx]
        offloading_portion = to_be_offloaded * offloading_decision
        self.task_mec_size[self.sbs_range.unsqueeze(-1), self.global_users_range, cur_tx_idx] += offloading_portion
        self.mec_arrival_buffer = self.mec_arrival_buffer * (1 - self.done_tx_tasks.float())

    def perform_mec_processing(self):
        # First, order each MEC task queue along the deadline dimension.
        # (Each BS has a queue of offloaded tasks from all devices along dim=1.)
        slot_indices = torch.arange(self.max_task_deadline + 1, device=self.device).reshape(1, 1, -1)
        slot_indices = slot_indices.expand(self.n_sbs, self.global_users, self.max_task_deadline + 1)
        slot_indices = (slot_indices + self.cur_mec_idx) % (self.max_task_deadline + 1)
        slots_ordered = torch.gather(self.task_mec_size, 2, slot_indices)

        # Find the first nonzero (i.e. non-empty) task in each BS×global_user queue
        condition = slots_ordered != 0
        next_task_idx = torch.argmax(condition.to(torch.int64), dim=2, keepdim=True)
        next_task_idx = ((next_task_idx + self.cur_mec_idx) % (self.max_task_deadline + 1))

        # Gather the current task sizes and corresponding deadlines.
        # (Shape: [n_sbs, global_users])
        cur_tasks = torch.gather(self.task_mec_size, 2, next_task_idx).squeeze(2)
        cur_tasks_deadline = torch.gather(self.task_mec_deadline, 2, next_task_idx).squeeze(2)

        # Identify which tasks were already “in processing” in the previous slot.
        common_idx = (next_task_idx.squeeze(2) == self.cur_mec_idx.squeeze(-1))

        # Separate the tasks that are already being processed (remained) from new ones.
        cur_tasks_remained = cur_tasks.clone()
        cur_tasks_remained[~common_idx] = 0  # keep only tasks that were already in service
        cur_tasks_new = cur_tasks.clone()
        cur_tasks_new[common_idx] = 0
        cur_tasks_deadline_new = cur_tasks_deadline.clone()
        cur_tasks_deadline_new[common_idx] = 0

        # ===== Determine which new tasks to start processing =====
        num_remained = (cur_tasks_remained > 0).sum(dim=1)  # shape: (n_sbs,)
        available_capacity = self.bs_config.es_capacity - num_remained
        available_capacity = torch.clamp(available_capacity, min=0)

        valid_new = cur_tasks_new > 0
        max_task_size = self.ue_config.max_task_size  # assume defined in your config
        huge_value = torch.tensor(1e9, device=self.device, dtype=cur_tasks_deadline_new.dtype)
        # Combined key: lower key = higher priority (tasks with lower deadline and/or size are preferred)
        combined_key = torch.where(valid_new,
                                   cur_tasks_deadline_new * (max_task_size + 1) + (max_task_size - cur_tasks_new),
                                   huge_value)
        sorted_indices = torch.argsort(combined_key, dim=1)
        ranks = torch.arange(self.global_users, device=self.device).unsqueeze(0).expand(self.n_sbs, -1)
        available_capacity_expanded = available_capacity.unsqueeze(1)
        selected_in_sorted = ranks < available_capacity_expanded
        selected_new = torch.zeros_like(selected_in_sorted, dtype=torch.bool)
        selected_new.scatter_(1, sorted_indices, selected_in_sorted)
        selected_new = selected_new & valid_new

        # ===== Combine the tasks that are already processing and the new ones selected =====
        processing_mask = (cur_tasks_remained > 0) | selected_new  # shape: (n_sbs, global_users)

        # --- NEW: Weighted power allocation based on the combined key ---
        # Instead of simply dividing power equally among tasks, we assign each processing task a weight
        # that is the inverse of its combined key (lower key means higher weight).
        epsilon = 1e-6
        fractions=self.task_mec_power_ratio.transpose(0, 2).transpose(1, 2).flatten(start_dim=1,
                                                                                                    end_dim=2)
        fraction=fractions[self.sbs_range.unsqueeze(-1), self.global_users_range, next_task_idx.squeeze()]
        # Total MEC computing power available per BS (assumed to be shaped [n_sbs, 1] originally)
        total_mec_power = self.mec_computing_power.squeeze(1)*self.bs_config.max_allocated_power_fraction  # shape: [n_sbs]
        # Allocate power to each processing task according to its weight fraction.
        allocated_power = fraction * total_mec_power.unsqueeze(1)  # shape: [n_sbs, global_users]

        # Compute the amount of task bits that can be processed in this slot per task.
        allocated_capacity = allocated_power * self.slot_duration/self.ue_config.task_power  # shape: [n_sbs, global_users]
        processed_amount = torch.where(
            processing_mask,
            torch.minimum(cur_tasks, allocated_capacity),
            torch.tensor(0.0, device=self.device)
        )
        new_remaining = cur_tasks - processed_amount

        # Compute the processing time and energy consumption for each task.
        processed_time = processed_amount * self.ue_config.task_power / (allocated_power + epsilon)
        power_watt = self.ue_config.task_power * self.bs_config.cpu_coefficient * (allocated_power ** 2)
        processed_energy = processed_time * power_watt  # in Joules
        # tot_energy = processed_energy.sum()

        # --- Update statistics and task queues (the following remains similar to your original code) ---
        total_processed = torch.zeros_like(slot_indices).float()
        total_processed[
            self.sbs_range.unsqueeze(-1), self.global_users_range, next_task_idx.squeeze()
        ] = processed_amount
        processed_grouped = total_processed.reshape((self.n_sbs, self.n_sbs, self.max_users, -1)) \
            .transpose(1, 2).transpose(0, 2)
        user_mec_processed = processed_grouped.sum(dim=2)
        self.task_mec_bits_done += user_mec_processed

        total_energy = torch.zeros_like(slot_indices).float()
        total_energy[
            self.sbs_range.unsqueeze(-1), self.global_users_range, next_task_idx.squeeze()
        ] = processed_energy
        total_energy_grouped = total_energy.reshape((self.n_sbs, self.n_sbs, self.max_users, -1)) \
            .transpose(1, 2).transpose(0, 2)
        user_mec_energy = total_energy_grouped.sum(dim=2)
        self.mec_computing_power_consumed += user_mec_energy

        # Finally, update the MEC task queue with the new remaining task sizes.
        self.task_mec_size[
            self.sbs_range.unsqueeze(-1),
            self.global_users_range,
            next_task_idx.squeeze(2)
        ] = torch.round(new_remaining, decimals=self.bs_config.decimal_round)

    def compute_local_delay(self) -> torch.Tensor:
        # If needed, flip the queue so that processing order is along dim=1 from index 0 onward.
        tasks = torch.flip(self.task_loc_size, dims=[-1])
        dlines = torch.flip(self.task_deadline, dims=[-1])

        # Compute processing times per task (each processing time is in time slots)
        # Use broadcasting if computing_power is a scalar.
        proc_times = torch.ceil(tasks * self.task_power / (self.slot_duration * self.ue_computing_power.unsqueeze(-1)))

        # Prepare a tensor for cumulative delay (one per user)
        num_sbs, num_users, queue_size = proc_times.shape
        # We'll accumulate along the queue dimension. We initialize with zero delay.
        cum_delay = torch.zeros(num_sbs, num_users, device=proc_times.device)

        # Loop over the queue dimension. (This loop is over queue_size only.)
        for i in range(queue_size):
            # For tasks that have not expired yet (i.e. waiting time is still less than the deadline),
            # add the processing time.
            valid = cum_delay < dlines[:, :, i]
            # Only add processing time where valid is True.
            cum_delay = cum_delay + proc_times[:, :, i] * valid.to(proc_times.dtype)

        # The final cum_delay is the delay to process the last task.
        # We return a [num_users, 1] tensor.
        return cum_delay
