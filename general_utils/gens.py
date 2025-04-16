import errno
import math
import os
import time
from logging import exception

import torch
import numpy as np
from pathlib import Path
import pickle
from matplotlib import font_manager, pyplot as plt




def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_results_directory(path_str) -> Path:
    if not Path(path_str).is_dir():
        Path(path_str).mkdir(parents=True)
    return Path(path_str)


def timeit(func):
    """
    Decorator for measuring function's running time.
    """

    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.2f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


def save_model(model, env):
    env_name = env.env_name
    glob_env_name = env.glob_env_name

    path_str = f'{get_project_root()}/saved_models/{glob_env_name}'
    if not Path(path_str).is_dir():
        Path(path_str).mkdir()
    torch.save(model.state_dict(), f'{path_str}/{env_name}.pt')
    print(f'wights are saved at {env_name}')
    par_uploaded = True
    with open(f'{path_str}/{env_name}.pkl', 'wb+') as f:
        pickle.dump([env.tot_users, par_uploaded], f)



def load_pretrained_actor_critic(env, policy, configs, is_offloading=False):
    env_name = env.env_name
    pretrained_env=env.trained_env
    env_str='offloading_' if is_offloading else 'noma_'
    actor_path_str = f'{get_project_root()}/saved_models/{pretrained_env}/{env_str+pretrained_env}{configs.chosen_env_idx}_actor.pt'
    critic_path_str = f'{get_project_root()}/saved_models/{pretrained_env}/{env_str+pretrained_env}{configs.chosen_env_idx}_critic.pt'
    if not Path(actor_path_str).is_file() or not Path(critic_path_str).is_file():
        print(f'Pretrained model not exist!')
        return
    actor_pars = torch.load(actor_path_str)
    critic_pars = torch.load(critic_path_str)
    print(f'Single-Checkpoint:Loading pretrained model at {env_str}{env_name}')
# offloading_policy.actor.hidden_layer.load_state_dict(actor_pars)
# offloading_policy.critic.hidden_layer.load_state_dict(critic_pars)
    policy.actor.load_state_dict(actor_pars)
    policy.critic.load_state_dict(critic_pars)
    print(f'pretrained weights are loaded at {env_str}{env_name}')


def load_pretrained_value_based_model(env, eval_net, target_net, configs, is_offloading=False):
    env_name = env.env_name
    pretrained_env=env.trained_env
    env_str='offloading_' if is_offloading else 'noma_'
    actor_path_str = f'{get_project_root()}/saved_models/{pretrained_env}/{env_str+pretrained_env}{configs.chosen_env_idx}_actor.pt'
    if not Path(actor_path_str).is_file():
        print(f'Pretrained model not exist!')
        return
    actor_pars = torch.load(actor_path_str)
    print(f'Single-Checkpoint:Loading pretrained model at {env_str}{env_name}')
    # offloading_policy.actor.hidden_layer.load_state_dict(actor_pars)
    # offloading_policy.critic.hidden_layer.load_state_dict(critic_pars)
    target_net.actor.load_state_dict(actor_pars)
    eval_net.actor.load_state_dict(actor_pars)
    print(f'pretrained weights are loaded at {env_str}{env_name}')


def load_pretrained_sac_model(env, agent, configs, is_offloading=False):
    env_name = env.env_name
    pretrained_env=env.trained_env
    env_str='offloading_' if is_offloading else 'noma_'
    chosen_env_name = env_str + pretrained_env + configs.chosen_env_idx
    model_path_str = f'{get_project_root()}/saved_models/{pretrained_env}/{chosen_env_name}'
    policy_path = model_path_str + '_policy_net.pt'
    q1_path = model_path_str + '_q_net1.pt'
    q2_path = model_path_str + '_q_net2.pt'
    target_q1_path = model_path_str + '_target_q_net1.pt'
    target_q2_path = model_path_str + '_target_q_net2.pt'
    if not Path(policy_path).is_file():
        print(f'Pretrained model not exist!')
        return
    policy_par=torch.load(policy_path)
    q1_par=torch.load(q1_path)
    q2_par=torch.load(q2_path)
    target_q1_par=torch.load(target_q1_path)
    target_q2_par=torch.load(target_q2_path)
    print(f'Single-Checkpoint:Loading pretrained model at {env_str}{env_name}')
    agent.policy_net.load_state_dict(policy_par)
    agent.q_net1.load_state_dict(q1_par)
    agent.q_net2.load_state_dict(q2_par)
    agent.target_q_net1.load_state_dict(target_q1_par)
    agent.target_q_net2.load_state_dict(target_q2_par)
    print(f'pretrained weights are loaded at {env_str}{env_name}')
