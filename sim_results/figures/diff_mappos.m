% 5 sbs, 4 sc, 3 sc capacity,  0.6 arrival rate, 0.3 type probabitly
% Define the list of algorithms, task arrival rates, and number of runs.
algo_list = {'proposed_dual_rnn_resnet';'proposed_dual_rnn_resnet_instant';'proposed_dual_plain';'proposed_dual_plain_instant';'mappo_joint_rnn_resnet';'mappo_joint_instant_rnn_resnet' }; % Extend as needed.
% algo_list = {'proposed'; 'maddpg';'madqn';'maddpg_device';'isac'}; % Extend as needed.

legend_names=["\textbf{Dual MAPPO (RNN + ResNet, L=10)}","\textbf{Dual MAPPO (RNN + ResNet, L=1)}","\textbf{Dual MAPPO (MLP, L=10)}","\textbf{Dual MAPPO (MLP, L=1)}","\textbf{Single MAPPO  (RNN + ResNet, L=10)}","\textbf{Single MAPPO  (RNN + ResNet, L=1)}"];
configs = [1];       % Example arrival rates.
n_runs = 10;  % Number of runs per algorithm & arrival rate.
show_less=true;
saved_path = '../data/test/diff_seeds/';
for_print=false;
load_print_info
load_common_step_pars

for m = 1:length(fl_metrics)
    % Reuse figure 'm'. If it exists, bring it forward and clear it.
    figure(m);
    clf;
    hold on;
    
    % Loop over each algorithm for this metric.
    for a = 1:length(algo_list)
        y_means = zeros(1, length(configs));
        y_ci = zeros(1, length(configs));
        
        % Loop over each task arrival rate.
        for i = 1:length(configs)
            run_values = zeros(1, n_runs);  % Preallocate run values
            
            % Loop over each run for the given algorithm and arrival rate.
            for j = 1:n_runs
                filename = sprintf('%s_best%d%s.mat', algo_list{a},  j, fl_metrics(m));
                full_file = fullfile(saved_path, filename);
                
                if exist(full_file, 'file')
                    data_struct = load(full_file);
                    if isfield(data_struct, 'result')
                        run_data = data_struct.result;
                    else
                        fields = fieldnames(data_struct);
                        run_data = data_struct.(fields{1});
                    end
                    run_values(j) = mean(run_data);
                end
            end
            
            % Compute the average and 95% confidence interval.
            avg_val = mean(run_values);
            std_val = std(run_values);
            t_val = tinv(0.975, n_runs-1);
            ci = t_val * std_val / sqrt(n_runs);
            
            y_means(i) = avg_val;
            y_ci(i) = ci;
        end
        
        % Plot error bars with custom style.
        errorbar(configs, y_means, y_ci, ...
            'LineStyle', lines{min(a, length(lines))}, ...
            'Marker', markers{min(a, length(markers))}, ...
            'Color', colors{min(a, length(colors))}, ...
            'LineWidth', line_size_print);
    end
    
    % Set labels, title, and legend with desired font size.
    xlabel("\boldmath $M$", 'Interpreter', 'latex', 'FontSize', font_size_print);
    ylabel(fl_labels(m), 'Interpreter', 'latex', 'FontSize', font_size_print);
    % title(sprintf('Performance Metric: %s', fl_labels(m)), 'Interpreter', 'latex', 'FontSize', font_size_print);
    legend(legend_names, 'Interpreter', 'latex', 'FontSize', font_size_print, 'Location', 'Best');
    grid on;
    set(gca, 'FontSize', font_size_print);
    hold off;
end
