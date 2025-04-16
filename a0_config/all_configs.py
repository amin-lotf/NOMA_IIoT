import json

import numpy as np

from general_utils import DotDic, gens
import torch

from general_utils.gens import get_results_directory

# torch.backends.cudnn.enabled = False

dir_name = 'a0_config'
configs_dir='configs'
global_loc = 'data'
e_config = DotDic(json.loads(open(f'{gens.get_project_root()}/general_configs/env_config.json', 'r').read()))

ue_config = DotDic(json.loads(open(f'{gens.get_project_root()}/general_configs/ue_config.json', 'r').read()))
bs_config = DotDic(json.loads(open(f'{gens.get_project_root()}/general_configs/bs_config.json', 'r').read()))
# fl_config = DotDic(json.loads(open(f'{gens.get_project_root()}/general_configs/task_config.json', 'r').read()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


configs = {
    'env_config': e_config,
    'bs_config': bs_config,
    'ue_config': ue_config,
    'device': device
}


def get_logging_config(loc):
    l_config = DotDic(json.loads(open(f'{gens.get_project_root()}/{dir_name}/logging_config.json', 'r').read()))
    l_config.results_dir = f"{l_config.pc_dir}{loc}"
    return l_config


def get_isac_config(add_to_seed, loc=None, is_testing=False):
    algo_name='isac'
    return  get_config(add_to_seed,algo_name,loc,is_testing)


def get_maddpg_config(add_to_seed, loc=None, is_testing=False):
    algo_name='maddpg'
    return  get_config(add_to_seed,algo_name,loc,is_testing)

def get_maddpg_device_config(add_to_seed, loc=None, is_testing=False):
    algo_name='maddpg_device'
    return  get_config(add_to_seed,algo_name,loc,is_testing)


def get_ppo_config(add_to_seed,loc=None,is_testing=False,adjuster=0):
    algo_name='ppo'
    return  get_config(add_to_seed,algo_name,loc,is_testing,adjuster=adjuster)

def get_ppo_joint_config(add_to_seed,loc=None,is_testing=False,adjuster=0):
    algo_name='ppo_joint'
    return  get_config(add_to_seed,algo_name,loc,is_testing,adjuster=adjuster)


def get_pso_config(add_to_seed,loc=None,is_testing=False):
    algo_name='pso'
    return  get_config(add_to_seed,algo_name,loc,is_testing)



def get_duel_ddqn_config(add_to_seed,loc=None,is_testing=False,adjuster=0):
    algo_name='duel_ddqn'
    return  get_config(add_to_seed,algo_name,loc,is_testing,adjuster=adjuster)



def get_config(add_to_seed,algo_name,loc=None,is_testing=False,adjuster=0):
    if loc is None:
        loc = global_loc
        if is_testing:
            loc+='/test/diff_seeds'
        else:
            loc+='/train/diff_seeds'

    alg_args = DotDic(
        json.loads(open(f'{gens.get_project_root()}/{algo_name}/{algo_name}_config.json', 'r').read()))
    alg_offloading_args = DotDic(
        json.loads(open(f'{gens.get_project_root()}/{algo_name}/{algo_name}_offloading_config.json', 'r').read()))
    configs[f'{algo_name}_config'] = alg_args
    configs[f'{algo_name}_offloading_config'] = alg_offloading_args

    base_seed = 12340
    rnd_seed = base_seed + add_to_seed
    if is_testing:
        if alg_args.use_single_checkpoint:
            trained_str = '_best'
        else:
            trained_str = '_avg'
        configs['trained_env'] = alg_args.env_name
        rnd_seed+=30
        configs['env_config'].testing_env = True
        configs[f'{algo_name}_config'].load_pretrained_weights = True
        configs[f'{algo_name}_offloading_config'].load_pretrained_weights = True
        configs[f'{algo_name}_offloading_config'].num_env_steps=2e4
        configs[f'{algo_name}_config'].num_env_steps=2e4

    elif alg_args.load_pretrained_weights or alg_offloading_args.load_pretrained_weights:
        trained_str= '_trained'
        configs['trained_env'] = alg_args.env_name
    else:
        trained_str = ''
        configs['trained_env']=None
    configs['glob_env_name'] = alg_args.env_name+trained_str
    configs['env_name'] =  alg_args.env_name+trained_str+str(add_to_seed + 1)
    alg_offloading_args['env_name']=configs['env_name']
    alg_offloading_args['n_envs'] = alg_args['n_envs']

    rnd_seed+=adjuster
    rng = np.random.default_rng(rnd_seed)
    torch.manual_seed(rnd_seed)


    configs['bs_config'].tot_users=rng.integers(configs['bs_config'].min_users, configs['bs_config'].max_users+1, size=1)[0]
    configs['logging_config'] = get_logging_config(loc)
    # rng = np.random.default_rng()
    configs['rng'] = rng
    return configs







