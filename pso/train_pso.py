from general_utils.custom_vec_simple_env import CustomVecSimpleEnv
from pso.pso_env import PSOEnv


def main(configs):
    all_args=configs['pso_config']

    # env = PPOEnv(configs)
    env_fns = [PSOEnv(configs) for _ in range(all_args.n_envs)]
    envs = CustomVecSimpleEnv(env_fns)
    from pso.runner.pso_runner import AlgoRunner as Runner

    runner = Runner(envs)
    runner.run(configs['logging_config'])



