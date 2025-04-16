#!/usr/bin/env python
from general_utils.custom_vec_env import CustomVecEnv
from ppo.ppo_env import PPOEnv


def main(configs):
    all_args=configs['ppo_config']
    # if all_args.algorithm_name == "rmappo":
    #     assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent offloading_policy!")
    # elif all_args.algorithm_name == "mappo":
    #     assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
    #         "check recurrent offloading_policy!")
    # else:
    #     raise NotImplementedError

    # env = PPOEnv(configs)
    env_fns = [PPOEnv(configs) for _ in range(all_args.n_envs)]
    envs = CustomVecEnv(env_fns)
    from ppo.runner.ppo_fl_runner import AlgoRunner as Runner

    runner = Runner(envs)
    runner.run(configs['logging_config'])



