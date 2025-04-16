
if show_less
% noma_metrics=["_step_reward","_step_m_rate","_step_pc","_step_num_failed_pdsc"];
noma_metrics=["_step_reward"];
noma_labels=["\textbf{Resource reward}";"\textbf{Average normalized data rate}";"\textbf{Average transmission power (dBm)}";"\textbf{Devices with failed PDSC}"];
xlabel_noma_name="\textbf{Time slot}";


fl_metrics=["_offloading_reward","_step_ep","_step_offloading_power","_step_bits_done","_step_drop_ratio","_step_sensitivity_delay"];
fl_labels=["\textbf{Offloading Reward}";"\textbf{EP (Mbits/J)}";"\textbf{ EC (J)}";"\textbf{Processed tasks (Mbits)}";"\textbf{Drop ratio}";"\textbf{Average delay (time slots)}"];
xlabel_fl_name="\textbf{Time slot}";
else
noma_metrics=["_step_reward","_step_num_failed_pdsc","_step_m_rate","_step_pc","_step_sum_rate"];
noma_labels=["\textbf{Step reward}";"\textbf{Devices with failed PDSC}";"\textbf{Average normalized data rate}";"\textbf{Average transmission power (dBm)}";"\textbf{Total data rate}"];
xlabel_noma_name="\textbf{Time slot}";


fl_metrics=["_offloading_reward","_step_ep","_step_delay","_step_offloading_power","_step_bits_done","_step_bits_balance","_step_drop_ratio","_step_sensitivity_delay","_step_mean_power","_step_mean_split"];
fl_labels=["\textbf{Step Reward}";"\textbf{EP}";"\textbf{Delay (time slots)}";"\textbf{Task power}";"\textbf{Processed tasks (Mb)}";"\textbf{Bits balance}";"\textbf{Drop ratio}";"\textbf{Sensitivity delay (time slots)}";"\textbf{Task mean power}";"\textbf{mean split}"];
xlabel_fl_name="\textbf{Time slot}";
end



% fl_metrics=["_fl_reward","_num_chosen_users","_num_legit_users","_test_accuracy","_test_loss","_train_loss","_training_latency","_acc_latency","_test_accuracy_norm"];

if for_print
    choose_every_noma=1;
    choose_every_fl=1;
    choose_every_fl_step=1;
    choose_every_fl_accuracy=1;
    choose_every_fl_full_train=1;
    
    k_points_noma=100;
    k_points_fl=100;
    k_points_fl_step=100;
    k_points_fl_accuracy=1;
    k_points_fl_full_train=1;
else
    choose_every_noma=1;
    choose_every_fl=1;
    choose_every_fl_step=1;
    choose_every_fl_accuracy=1;
    choose_every_fl_full_train=1;
    
    k_points_noma=1;
    k_points_fl=1;
    k_points_fl_step=100;
    k_points_fl_accuracy=1;
    k_points_fl_full_train=1;
end



noma_with_limit=true;
fl_with_limit=true;
fl_step_with_limit=true;
fl_accuracy_with_limit=true;
fl_full_with_limit=true;

