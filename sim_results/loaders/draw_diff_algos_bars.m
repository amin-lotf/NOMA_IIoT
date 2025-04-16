load_info
metrics_list=[];
next_fig=0;
% save_dir="all_diff_steps";


load_common_mean_pars


metrics_type={noma_metrics_m;fl_metrics_m};
metrics_y_axis_labels={noma_labels;fl_labels};
metricx_x_axis_labels=[xlabel_noma_name,xlabel_fl_name];


for type_idx=1:size(metrics_type,1)
    next_fig=next_fig+size(metrics_list,2);
    metrics_list=metrics_type{type_idx};
    load_diff_algos_mean_data
    tot_figures=cell(size(metrics_list,2));
    x_axis_label=metricx_x_axis_labels(type_idx);
    y_axis_labels=metrics_y_axis_labels{type_idx};
    for i=1:size(metrics_list,2)
        metric=metrics_list(i);
        tot_figures{i}=figure(next_fig+i);
        clf(tot_figures{i},'reset')
        tot_metric_name=strcat('tot',metric);
        tot_data=eval(tot_metric_name);
        num_runs=size(tot_data,1);
        bar_legend_names=categorical(legend_names(1:num_runs));
        bar(legend_names(1:num_runs),tot_data)
        hold on;
        
        xlabel({x_axis_label},'Interpreter','latex','FontSize', font_size_print);
        ylabel({y_axis_labels(i)},'Interpreter','latex','FontSize', font_size_print);
        set(gca,'FontSize',font_size_print)
        grid on
        % xlim([0,min([max_x,min_size])])
        set(gca,'TickLabelInterpreter','latex')
    end
end