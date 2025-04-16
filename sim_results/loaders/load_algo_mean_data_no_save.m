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

all_data={};
run_counter=0;
% fprintf('%s:  ',path_to_algo);
for run=1:max_runs
    path_to_algo_run=strcat(path_to_algo,int2str(run));
    target_data=strcat(path_to_algo_run,metric,".mat");
    if isfile(target_data)
        a=load(target_data);
        b=struct2cell(a);
        c=b{1};
        
        % metric_name=strcat(algo_name,int2str(run),metric);
        % assignin('base',metric_name,c)
        
        if c>0
            all_data{run_counter+1}=c;
            run_counter=run_counter+1;
        % else
        %     fprintf(' %d',run)
        end
    else
        
    end          
end

tot_metric_name=strcat(algo_name,'_tot',metric);
% if run_counter>0 & run_counter == max_runs
if run_counter>0
    result_matrix=zeros(run_counter,1);
    for i=1:run_counter
        result_matrix(i,1)=all_data{i}(1);
    end
    mean_data=mean(result_matrix,1);
else
    % fprintf('No data yet!')
    mean_data=0;
end
% fprintf('\n');
mean_metric_name=strcat(algo_name,'_mean',metric);
assignin('base',mean_metric_name,mean_data)
