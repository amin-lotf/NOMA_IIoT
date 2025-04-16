#!/usr/bin/env python
from general_utils.custom_vec_env import CustomVecEnv
# from maddpg.maddpg_env import MADDPGEnv
from maddpg_device.maddpg_env import MADDPGEnv


def main(configs):
    all_args=configs['maddpg_device_config']
    # if all_args.algorithm_name == "rmamaddpg":
    #     assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent offloading_policy!")
    # elif all_args.algorithm_name == "mamaddpg":
    #     assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
    #         "check recurrent offloading_policy!")
    # else:
    #     raise NotImplementedError

    # env = maddpgEnv(configs)
    env_fns = [MADDPGEnv(configs) for _ in range(all_args.n_envs)]
    envs = CustomVecEnv(env_fns)
    from maddpg_device.runners.maddpg_runner import AlgoRunner as Runner

    runner = Runner(envs)
    runner.run(configs['logging_config'])



