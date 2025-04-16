for metric=metrics_list
        comp_data=zeros(size(algo_list,1),size(all_paths,1));
        for alg_idx=1:size(algo_list,1)
            algo_name=algo_list{alg_idx};
            alg_res_matrix=zeros(size(all_paths,1),1);
            path_to_algo=strcat(path_to_algo,algo_name);
            % save_mean_path=all_paths_mean(path_idx);
            load_algo_step_data_no_save
            % path_to_mean=strcat(all_paths_mean(path_idx),algo_name,'_mean',comp_metric,".mat");
            mean_metric_name=strcat(algo_name,'_mean',metric);
            c=eval(mean_metric_name);
            % b=struct2cell(a);
            % c=b{1};
            alg_res_matrix(path_idx)=c;
            comp_data(alg_idx,path_idx)=c;
            % metric_name=strcat(cur_algo,'_mean',metric);
            % assignin('base',metric_name,c)
            % run_counter=run_counter+1;
                
   
        end
        % result_matrix=zeros(run_counter,1);
        % for i=1:run_counter
        %     result_matrix(i,1)=all_data{i}(1);
        % end
        tot_metrix_name=strcat('comp',metric);
        assignin('base',tot_metrix_name,comp_data)
end
