for metric=metrics_list
        all_data=zeros(size(algo_list,1),size(all_paths,1));
        run_counter=0;
        for run=1:size(algo_list,1)
            cur_algo=algo_list{run};
            alg_res_matrix=zeros(size(all_paths,1),1);
            for path_idx=1:size(all_paths,1)
                path_to_algo=strcat(all_paths(path_idx),cur_algo,'_mean',metric,".mat");
                if isfile(path_to_algo)
                    a=load(path_to_algo);
                    b=struct2cell(a);
                    c=b{1};
                    alg_res_matrix(path_idx)=c;
                    all_data(run,path_idx)=c;
                    % metric_name=strcat(cur_algo,'_mean',metric);
                    % assignin('base',metric_name,c)
                    % run_counter=run_counter+1;
                end
            end
        end
        % result_matrix=zeros(run_counter,1);
        % for i=1:run_counter
        %     result_matrix(i,1)=all_data{i}(1);
        % end
        tot_metrix_name=strcat('comp',metric);
        assignin('base',tot_metrix_name,all_data)
end
