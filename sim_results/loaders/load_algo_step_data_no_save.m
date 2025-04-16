% input data:
%   algo_name
%save_dir
%   max_runs
%   noma_metrics
%   fl_metrics 
%   k_points_noma,
%   k_points_fl,



all_data={};
run_counter=0;
if with_limit
    min_size=1000000;
else
    max_size=0;
end
% fprintf('%s:  ',path_to_algo);
ok_matrix=zeros(max_runs,1);
for run=1:max_runs
    path_to_algo_run=strcat(path_to_algo,int2str(run));
    target_data=strcat(path_to_algo_run,metric,".mat");
    if isfile(target_data)
        a=load(target_data);
        b=struct2cell(a);
        c=b{1};
        
        % metric_name=strcat(algo_name,int2str(run),metric);
        % assignin('base',metric_name,c)

        if sum(c)>0
            data_size=size(c,2);
            if with_limit
                min_size=min(min_size,data_size);
            else
                max_size=max(max_size,data_size);
            end
            c = movmean(c,k_point);
            all_data{run}=c;
            metric_name=strcat(algo_name,int2str(run),metric);
            assignin('base',metric_name,c)
            run_counter=run_counter+1;
            ok_matrix(run)=1;

            % else
            %     fprintf(' %d',run)
        else
            all_data{run}=0;
        end
    else
        all_data{run}=0;
    end
end
if with_limit
    result_matrix=zeros(run,min_size);
    mean_matrix=zeros(run_counter,min_size);
    tmp_counter=1;
    for i=1:max_runs
        if ok_matrix(i)==1
            result_matrix(i,:)=all_data{i}(1:min_size);
            mean_matrix(tmp_counter,:)=all_data{i}(1:min_size);
            tmp_counter=tmp_counter+1;
        end
    end
else
    result_matrix=zeros(run,max_size);
    mean_matrix=zeros(run_counter,max_size);
    tmp_counter=1;
    for i=1:max_runs
        if ok_matrix(i)==1
            result_matrix(i,:)=all_data{i}(1:size(all_data{i},2));
            mean_matrix(tmp_counter,:)=all_data{i}(1:size(all_data{i},2));
            tmp_counter=tmp_counter+1;
        end
    end
end

% tot_metric_name=strcat(algo_name,'_tot',metric);
% assignin('base',tot_metric_name,result_matrix)
mean_data=mean(mean_matrix,1);
mean_metric_name=strcat(algo_name,'_mean',metric);
assignin('base',mean_metric_name,mean_data)
% ok_metric_name=strcat(algo_name,'_ok',metric);
% assignin('base',mean_metric_name,ok_matrix)