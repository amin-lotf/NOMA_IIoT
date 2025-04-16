% input data:
%   algo_name
%save_dir
%   max_runs
%   noma_metrics
%   fl_metrics 
%   k_points_noma,
%   k_points_fl,
path_to_algo=strcat('data/',algo_name);
save_mean_path=strcat('../../',save_dir,'/data/');
k_points=[k_points_noma,k_points_fl,k_points_fl_step];
all_metrics={metrics_list};
for metric_idx=1:size(all_metrics,2)
    metric_type=all_metrics{metric_idx};
    for metric=metric_type
            all_data={};
            run_counter=0;
            min_size=1000000;
            for run=1:max_runs
                path_to_algo_run=strcat(path_to_algo,int2str(run));
                target_data=strcat(path_to_algo_run,metric,".mat");
                if isfile(target_data)
                    a=load(target_data);
                    b=struct2cell(a);
                    c=b{1};
                    data_size=size(c,2);
                    min_size=min(min_size,data_size);
                    c = movmean(c,k_points(metric_idx));
                    % c=c+c*0.15;
                    all_data{run}=c;
                    metric_name=strcat(algo_name,int2str(run),metric);
                    assignin('base',metric_name,c)
                    run_counter=run_counter+1;
                end          
            end
            result_matrix=zeros(run_counter,min_size);
            for i=1:run_counter
                result_matrix(i,:)=all_data{i}(1:min_size);
            end
            tot_metric_name=strcat(algo_name,'_tot',metric);
            assignin('base',tot_metric_name,result_matrix)
            mean_data=mean(result_matrix,1);
            mean_metric_name=strcat(algo_name,'_mean',metric);
            assignin('base',mean_metric_name,mean_data)
            save_to=strcat(save_mean_path,mean_metric_name,'.mat');
            fname = sprintf(save_to, mean_metric_name); 
            save(fname,mean_metric_name)
    end
end

