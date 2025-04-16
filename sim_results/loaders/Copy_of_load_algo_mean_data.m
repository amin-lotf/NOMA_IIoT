% input data:
%   algo_name
%save_dir
%   max_runs
%   noma_metrics
%   fl_metrics 
%   k_points_noma,
%   k_points_fl,

if ~exist(save_mean_path, 'dir')
    mkdir(save_mean_path); % Create the folder recursively
    disp(['Folder created: ' save_mean_path]);
% else
%     disp(['Folder already exists: ' save_mean_path]);
end

metric_types={noma_metrics_m,fl_metrics_m};
for metric=metrics_list
    all_data={};
    run_counter=0;
    for run=1:max_runs
            path_to_algo_run=strcat(path_to_algo,int2str(run));
            target_data=strcat(path_to_algo_run,metric,".mat");
            if isfile(target_data)
                a=load(target_data);
                b=struct2cell(a);
                c=b{1};
                all_data{run}=c;
                metric_name=strcat(algo_name,int2str(run),metric);
                assignin('base',metric_name,c)
                run_counter=run_counter+1;
            end          
    end
    result_matrix=zeros(run_counter+1,1);
    for i=1:run_counter
        result_matrix(i+1,1)=all_data{i}(1);
    end
    tot_metric_name=strcat(algo_name,'_tot',metric);
    if run_counter>0
        mean_data=mean(result_matrix(2:run_counter+1),1);
    else
        mean_data=0;
    end
    result_matrix(1,1)=mean_data;
    assignin('base',tot_metric_name,result_matrix)
    mean_metric_name=strcat(algo_name,'_mean',metric);
    assignin('base',mean_metric_name,mean_data)
    save_to=strcat(save_mean_path,mean_metric_name,'.mat');
    fname = sprintf(save_to, mean_metric_name); 
    save(fname,mean_metric_name)
end
