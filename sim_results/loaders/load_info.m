save_image=false;
font_size=12;
marker_size=8;
slot_duration=0.1;
marker_size_m=6;
line_size = 1.8;
line_size_print = 0.5;%1.5;
font_size_print=6;%12;
font_size_print_m=12;
font_size_reward=12;
font_size_outer = 6;
fig_width = 5.6; % Single column width in inches
% fig_width = 7; % Double column width in inches
fig_height = 4.2; % Choose an appropriate height in inches

label_test_accuracy='\textbf{Accuracy}';
label_test_loss='\textbf{Loss}';
label_training_latency='\textbf{Training latency}';
label_legit_ratio='\textbf{Ratio of successful participation}';
label_step_m_rate='\textbf{Average data rate (Mbps)}';
label_step_num_failed_pdsc='\textbf{Devices with failed PDSC}';
label_w_mbit='\textbf{Processed data (Mbits)}';
label_delay='\textbf{Average delay (time slots)}';
label_rate='\textbf{Sum rate (Mbps)}';


random_device_name='\textbf{Random Device Selection}';
all_device_name='\textbf{Full Participation}';
duel_ddqn_name='\textbf{3DQN-Full Participation}';

duel_ddqn_marker = '-o';
mappo_no_tx_marker = '-s';
random_device_marker = '-s';
all_device_marker = '-v';
ma3mco_marker = '-d';
isac_marker = '->';

duel_ddqn_marker_radar = 'o';
mappo_no_tx_marker_radar = 's';
random_device_marker_radar = 's';
all_device_marker_radar = 'v';
ma3mco_marker_radar = 'd';
isac_marker = '->';

markers=[ "o";"s";"v";"d";"*";"+";"square";"^";"o";"s";"v";"d";"->";"*";"+";"square";"^";"o";"s";"v";"d";"->";"*";"+";"square";"^";"o"];
lines=["-";"-";"-";"-.";":";"-";"--";"-.";":";"-";"--";"-";"-";"-";"-.";":";"-";"--";"-.";":";"-";"--";"-";"-";"-";"-.";":";"-";"--";"-.";":";"-";"--"];
colors=["#000000";"#D95319";"#EDB120";"#7E2F8E";"#4DBEEE";"#0072BD";"#D95319";"#EDB120";"#7E2F8E";"#4DBEEE";"#D95319";"#EDB120";"#7E2F8E";"#4DBEEE";"#0072BD";"#D95319";"#EDB120";"#7E2F8E";"#4DBEEE";"#D95319";"#EDB120";"#7E2F8E";"#4DBEEE";"#0072BD";"#D95319";"#EDB120";"#7E2F8E";"#4DBEEE"];

duel_ddqn_color='#0072BD';
mappo_nzo_tx_color='#D95319';
random_device_color='#D95319';
all_device_color='#EDB120';
ma3mco_color='#7E2F8E';
isac_color='#4DBEEE';

duel_ddqn_line='-';
mappo_no_tx_line='--';
random_device_line = '-';
all_device_line = '-.';
ma3mco_line = '--';
isac_line = ':';
