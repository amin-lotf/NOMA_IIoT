% input data:
%   algo_list
%   noma_metrics
%   fl_metrics 
%   k_points_noma,
%   k_points_fl,
metric_types={noma_metrics,fl_metrics,fl_step_metrics};
k_points=[k_points_noma,k_points_fl,k_points_fl_step];
for metric_idx=1:size(metric_types,2)
    metric_type=metric_types{metric_idx};
    for metric=metric_type
            all_data={};
            run_counter=0;
            min_size=1000000;
            for run=1:size(algo_list,1)
                cur_algo=algo_list{run};
                path_to_algo=strcat('data/',cur_algo,'_mean',metric,".mat");
                if isfile(path_to_algo)
                    a=load(path_to_algo);
                    b=struct2cell(a);
                    c=b{1};
                    data_size=size(c,2);
                    min_size=min(min_size,data_size);
                    c = movmean(c,k_points(metric_idx));
                    all_data{run}=c;
                    metric_name=strcat(cur_algo,'_mean',metric);
                    assignin('base',metric_name,c)
                    run_counter=run_counter+1;
                end          
            end
            result_matrix=zeros(run_counter,min_size);
            for i=1:run_counter
                result_matrix(i,:)=all_data{i}(1:min_size);
            end
            tot_metric_name=strcat('tot',metric);
            assignin('base',tot_metric_name,result_matrix)
    end
end

