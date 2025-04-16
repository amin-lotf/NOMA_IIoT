max_runs=10;
algo_name="proposed_deadline_05";

path_to_algo=strcat('../data/train/diff_seeds/',algo_name);
save_mean_path=strcat('../data/train/mean_data/');
with_mean=true;
for_print=true;
with_marker=false;
max_x_size=600000;
show_less=true;
% legend_names=["Run1";"Run2";"Run3";"Run4";"Run5";"Run6";"Run7";"Run8";"Run9"];
legend_names=["Mean";"BS1";"BS2";"BS3";"BS4";"BS5";"BS6";"BS7";"BS8";"BS9";"BS10"];

draw_algo_figures



