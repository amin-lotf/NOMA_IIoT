#!/usr/bin/env python
from general_utils.custom_vec_env import CustomVecEnv
from isac.sac_env import SACEnv


def main(configs):
    all_args = configs['isac_config']
    env_fns = [SACEnv(configs) for _ in range(all_args.n_envs)]
    envs = CustomVecEnv(env_fns)
    from isac.runner.sac_runner import AlgoRunner as Runner

    runner = Runner(envs)
    runner.run(configs['logging_config'])
