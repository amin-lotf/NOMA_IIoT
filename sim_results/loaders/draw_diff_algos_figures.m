if for_print
    load_print_info
else
    load_info
end
metrics_list=[];
next_fig=0;


load_common_step_pars

fl_accuracy_with_limit=false;

metrics_type={noma_metrics;fl_metrics};
metrics_y_axis_labels={noma_labels;fl_labels};
metricx_x_axis_labels=[xlabel_noma_name,xlabel_fl_name];
choose_every=[choose_every_noma,choose_every_fl];
key_points=[k_points_noma,k_points_fl];
with_limits=[noma_with_limit,fl_with_limit];


for type_idx=1:size(metrics_type,1)
    next_fig=next_fig+size(metrics_list,2);
    metrics_list=metrics_type{type_idx};
    k_point=key_points(type_idx);
    with_limit=with_limits(type_idx);
    load_diff_algos_data
    tot_figures=cell(size(metrics_list,2));
    x_axis_label=metricx_x_axis_labels(type_idx);
    y_axis_labels=metrics_y_axis_labels{type_idx};
    choose_every_step=choose_every(type_idx);
    for i=1:size(metrics_list,2)
        metric=metrics_list(i);
        tot_metric_name=strcat('tot',metric);
        tot_data=eval(tot_metric_name);
        min_size=size(tot_data,2);
        if with_marker
            min_size=min(min_size,max_x_size-1);
        else
            min_size=min(min_size,max_x_size+1);
        end
        
        tot_figures{i}=figure(next_fig+i);
        clf(tot_figures{i},'reset')
        if ~with_limit
            limit_metric_name=strcat('limit',metric);
            limit_vals=eval(limit_metric_name);
        end
        
    
        
        num_runs=size(tot_data,1);
        for run=1:num_runs
            if with_limit
                indices = 1:choose_every_step:min_size;
                
            else
                indices = 1:choose_every_step:min(min_size,limit_vals(run,1));
            end
            if with_marker
                    plot(indices, tot_data(run,indices), markers(run+1), 'Color',  colors(run+1), 'LineStyle',lines(run+1),'Linewidth', line_size_print,  'MarkerSize', marker_size)
                    hold on;
            else
                    plot(indices, tot_data(run,indices),  'Color',  colors(run+1), 'Linewidth', line_size_print,  'MarkerSize', marker_size)
                    hold on;
            end
        end
        legend(legend_names,'Interpreter','latex','Location','northoutside','FontSize', font_size_outer,'NumColumns',1);
        xlabel({x_axis_label},'Interpreter','latex','FontSize', font_size_print);
        ylabel({y_axis_labels(i)},'Interpreter','latex','FontSize', font_size_print);
        set(gca,'FontSize',font_size_print)
        ax = gca;
        ax.XAxis.Exponent = 3;
        grid on
        xlim([0,min([min_size,max_x_size])+1])
        
        % xlim([0,min([max_x,min_size])])
        set(gca,'TickLabelInterpreter','latex')
    end
end