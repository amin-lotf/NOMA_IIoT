save_image=false;
font_size=12;
marker_size=6;
slot_duration=0.1;
marker_size_m=3;
line_size = 1.8;
line_size_print = 2.2;%1.5;
font_size_print=10.5;%12;
font_size_print_m=12;
font_size_reward=12;
font_size_outer= 9.5;
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
lines = [
    "-";
    "--";
    "--";
    ":";
    "-";
    "-.";
];


colors = [
    "#000000";  % Black - Neutral, always useful
    "#0072BD";  % Dark Blue - Professional and clear
    "#D95319";  % Burnt Orange - Distinct and non-bright
    "#4DBEEE";  % Teal - Soft and pleasant
    "#7E2F8E";  % Purple - Deep and distinguishable
    "#A2142F";  % Dark Red - Avoids brightness, adds variety
    "#EDB120";  % Muted Gold - Retained but not too bright
    "#2E8B57";  % Sea Green - Calm and clear
    "#5E5E5E";  % Dark Gray - Good for subtle differences
    "#8B4513";  % Saddle Brown - Earthy and distinct
    "#6B8E23";  % Olive Green - Soft and journal-friendly
    "#4682B4";  % Steel Blue - Not too bright and very clear
    "#B8860B";  % Dark Goldenrod - Non-bright earthy tone
    "#4169E1";  % Royal Blue - Professional and clear
    "#A0522D";  % Sienna - Another earthy, muted color
];

colors = [
    "#4C4C4C";
    "#606060";
    "#737373";
    "#8C8C8C";
    "#A6A6A6";
    "#BFBFBF";
    "#D9D9D9"
];


colors = [
    "#0072B2";  % Blue
    "#D55E00";  % Vermilion
    "#009E73";  % Bluish Green
    "#E69F00";  % Orange
    % "#F0E442";  % Yellow
    "#4DBEEE";
    "#CC79A7";  % Reddish Purple
    "#56B4E9";   % Sky Blue
    "#A0522D";  % Sienna - Another earthy, muted color
    "#4169E1";  % Royal Blue - Professional and clear
];


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
