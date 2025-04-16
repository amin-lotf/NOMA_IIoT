% input data:
%   algo_list
%   noma_metrics
%   fl_metrics 
%   k_points_noma,
%   k_points_fl,

for metric=metrics_list
        all_data={};
        run_counter=0;
        if with_limit
        min_size=1000000;
        else
            max_size=0;
        end
        for run=1:size(algo_list,1)
            cur_algo=algo_list{run};
            path_to_algo=strcat(path_to_mean,cur_algo,'_mean',metric,".mat");
            if isfile(path_to_algo)
                a=load(path_to_algo);
                b=struct2cell(a);
                c=b{1};
                data_size=size(c,2);
                if with_limit
                    min_size=min(min_size,data_size);
                else
                    max_size=max(max_size,data_size);
                end
                c = movmean(c,k_point);
                all_data{run}=c;
                metric_name=strcat(cur_algo,'_mean',metric);
                assignin('base',metric_name,c)
                run_counter=run_counter+1;
            end          
        end
        if with_limit
            result_matrix=zeros(run_counter,min_size);
            for i=1:run_counter
                result_matrix(i,:)=all_data{i}(1:min_size);
            end
        else
            result_matrix=zeros(run_counter,max_size);
            limit_matrix=zeros(run_counter,1);
            for i=1:run_counter
                result_matrix(i,1:size(all_data{i},2))=all_data{i}(1:size(all_data{i},2));
                limit_matrix(i,1)=size(all_data{i},2);
            end
            limit_metrix_name=strcat('limit',metric);
            assignin('base',limit_metrix_name,limit_matrix)
        end
        tot_metrix_name=strcat('tot',metric);
        assignin('base',tot_metrix_name,result_matrix)      
end


