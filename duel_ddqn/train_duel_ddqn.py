#!/usr/bin/env python
import numpy as np
import torch
from duel_ddqn.duel_ddqn_env import DuelDDQNEnv
from duel_ddqn.runner.duel_ddqn_runner import AlgoRunner as Runner
from general_utils.custom_vec_env_stateless import CustomVecEnvStateless


def main(configs):
    all_args = configs['duel_ddqn_config']
    env_fns = [DuelDDQNEnv(configs) for _ in range(all_args.n_envs)]
    envs = CustomVecEnvStateless(env_fns)

    runner = Runner(envs)
    runner.run(configs['logging_config'])
