for metric=metrics_list
        all_data={};
        run_counter=0;
        for run=1:size(algo_list,1)
            cur_algo=algo_list{run};
            path_to_algo=strcat(path_to_mean,cur_algo,'_mean',metric,".mat");
            if isfile(path_to_algo)
                a=load(path_to_algo);
                b=struct2cell(a);
                c=b{1};
                all_data{run}=c;
                metric_name=strcat(cur_algo,'_mean',metric);
                assignin('base',metric_name,c)
                run_counter=run_counter+1;
            end
        end
        result_matrix=zeros(run_counter,1);
        for i=1:run_counter
            result_matrix(i,1)=all_data{i}(1);
        end
        tot_metrix_name=strcat('tot',metric);
        assignin('base',tot_metrix_name,result_matrix)
end
