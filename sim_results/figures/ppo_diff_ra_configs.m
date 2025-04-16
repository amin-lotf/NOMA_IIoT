% 4 sbs, 4 sc, 3 sc capacity, 0.6 arrival rate, 0.3 type probabitly

% algo_list={"proposed_ra_config_1";"proposed_ra_config_2";"proposed_ra_config_3";"proposed_ra_config_4";"proposed_ra_config_5";"proposed_ra_config_6";"proposed_ra_config_7";"proposed_ra_config_8";"proposed_ra_config_9";"proposed_ra_config_10"};
algo_list={"proposed_ra_config_1";"proposed_ra_config_2";"proposed_ra_config_3";"proposed_ra_config_4";"proposed_ra_config_5"};
% algo_list={"proposed_offloading_config1_trained";"proposed_offloading_config3_trained";"proposed_offloading_config2_trained";"proposed_offloading_config4_trained"};


% legend_names=["Run1";"Run2";"Run3";"Run4";"Run5";"Run6";"Run7";"Run8";"Run9"];
legend_names=[
    ' \boldmath$\textbf{actor\_lr} =1\times 10^{-5}$, \boldmath$\textbf{critic\_lr} =2\times 10^{-5}$, \boldmath$ T_{ra}=128$, $\phantom{8}$\boldmath$L=10$, \boldmath$E_{ra}=10$';
    ' \boldmath$\textbf{actor\_lr} =1\times 10^{-5}$, \boldmath$\textbf{critic\_lr} =2\times 10^{-5}$, \boldmath$ T_{ra}=256$, $\phantom{8}$\boldmath$L=10$, \boldmath$E_{ra}=10$';
    ' \boldmath$\textbf{actor\_lr} =3\times 10^{-5}$, \boldmath$\textbf{critic\_lr} =4\times 10^{-5}$, \boldmath$ T_{ra}=128$,$\phantom{8}$ \boldmath$L=5$,$\phantom{8}$   \boldmath$E_{ra}=5$';
    ' \boldmath$\textbf{actor\_lr} =3\times 10^{-5}$, \boldmath$\textbf{critic\_lr} =4\times 10^{-5}$, \boldmath$ T_{ra}=128$,$\phantom{8}$   \boldmath$L=5$,$\phantom{8}$  \boldmath$E_{ra}=10$';
    ' \boldmath$\textbf{actor\_lr} =3\times 10^{-5}$, \boldmath$\textbf{critic\_lr} =4\times 10^{-5}$, \boldmath$ T_{ra}=128$,$\phantom{8}$  \boldmath$L=10$,  \boldmath$E_{ra}=10$';
    ' \boldmath$\textbf{actor\_lr} =5\times 10^{-5}$, \boldmath$\textbf{critic\_lr} =7\times 10^{-5}$, \boldmath$ T_{ra}=128$,$\phantom{8}$  \boldmath$L=10$, \boldmath$E_{ra}=10$';
    ' \boldmath$\textbf{actor\_lr} =5\times 10^{-5}$, \boldmath$\textbf{critic\_lr} =7\times 10^{-5}$, \boldmath$ T_{ra}=64$,$\phantom{8}$  \boldmath$L=10$, \boldmath$E_{ra}=10$';
    ' \boldmath$\textbf{actor\_lr} =1\times 10^{-4}$, \boldmath$\textbf{critic\_lr} =2\times 10^{-4}$, \boldmath$ T_{ra}=128$,$\phantom{8}$  \boldmath$L=15$, \boldmath$E_{ra}=15$';
    ' \boldmath$\textbf{actor\_lr} =1\times 10^{-4}$, \boldmath$\textbf{critic\_lr} =2\times 10^{-4}$, \boldmath$ T_{ra}=64$,$\phantom{8}$  \boldmath$L=10$, \boldmath$E_{ra}=10$';
    ' \boldmath$\textbf{actor\_lr} =3\times 10^{-4}$, \boldmath$\textbf{critic\_lr} =4\times 10^{-4}$, \boldmath$ T_{ra}=64$,$\phantom{8}$  \boldmath$L=10$, \boldmath$E_{ra}=10$';
    "BS7";"BS8";"BS9";"BS10"];

% legend_names=[
%     ' \boldmath$\textbf{actor\_lr} =1\times 10^{-5}$, \boldmath$\textbf{critic\_lr} =2\times 10^{-5}$, \boldmath$ T_{to}=128$, $\phantom{8}$\boldmath$L=20$, \boldmath$E_{to}=10$';
%     ' \boldmath$\textbf{actor\_lr} =2\times 10^{-5}$, \boldmath$\textbf{critic\_lr} =3\times 10^{-5}$, \boldmath$ T_{to}=64$,$\phantom{8}\;\,$   \boldmath$L=20$, \boldmath$E_{to}=10$';
%     ' \boldmath$\textbf{actor\_lr} =3\times 10^{-5}$, \boldmath$\textbf{critic\_lr} =5\times 10^{-5}$, \boldmath$ T_{to}=128$,$\phantom{8}$ \boldmath$L=10$, \boldmath$E_{to}=10$';
%     ' \boldmath$\textbf{actor\_lr} =5\times 10^{-5}$, \boldmath$\textbf{critic\_lr} =7\times 10^{-5}$, \boldmath$ T_{to}=256$,$\phantom{8}$ \boldmath$L=20$, \boldmath$E_{to}=5$';
%     ' \boldmath$\textbf{actor\_lr} =1\times 10^{-4}$, \boldmath$\textbf{critic\_lr} =2\times 10^{-4}$, \boldmath$ T_{to}=64$,$\phantom{8}\;$  \boldmath$L=10$, \boldmath$E_{to}=5$';
%     ' \boldmath$\textbf{actor\_lr} =1\times 10^{-4}$, \boldmath$\textbf{critic\_lr} =2\times 10^{-4}$, \boldmath$ T_{to}=128$,\boldmath$L=20$, \boldmath$E_{to}=10$';
%     "BS7";"BS8";"BS9";"BS10"];


show_less=false;
with_mean=true;
for_print=true;
with_marker=false;
max_x_size=140000;
path_to_mean=strcat('../data/train/mean_data/');

draw_diff_algos_figures



