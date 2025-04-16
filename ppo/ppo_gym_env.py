import numpy as np
import torch
from gymnasium import spaces
import  gymnasium as gym
from env_models.mec_environment import MECEnvironment
from pettingzoo.classic import tictactoe_v3
from pettingzoo.sisl import multiwalker_v9


class PPOGymEnv:
    def __init__(self, configs):
        self.env_name = configs['env_name']
        self.glob_env_name = configs['glob_env_name']
        self.trained_env = configs['trained_env']
        self.device = configs['device']
        self.ppo_tx_config = configs['ppo_config']

        # gym_env = gym.make("CartPole-v1")
        # gym_env = gym.make('Ant-v4', ctrl_cost_weight=0.5)
        # gym_env = gym.make("MountainCarContinuous-v0")
        # gym_env = gym.make("LunarLander-v2", continuous=True)
        # env = tictactoe_v3.env()
        env = multiwalker_v9.parallel_env(n_walkers=3)
        self.num_tx_agents = 3
        self.gym_env=env
        # self.agents= sorted(env.agents)
        self.total_reward = 0
        # if gym_env.action_space.__class__.__name__==
        self.action_mask = np.ones((self.num_tx_agents, 1), dtype=np.int64)
        self.reset()



    def get_available_action_shape(self,action_type=0):
        return self.action_mask.shape[1]


    def get_observation_space(self,observation_type=0):
        return self.gym_env.observation_space(self.gym_env.possible_agents[0])

    def get_state_space(self,state_type=0):
        obs_dim = self.get_observation_space().shape[0]
        return spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim*3,), dtype=np.float32)

    def get_action_space(self,action_type=0):
        return self.gym_env.action_space(self.gym_env.possible_agents[0])

    def reset(self, seed=None, options=None,step_type=0):
        # state = torch.tensor(self.gym_env.reset()[0]).to(self.device).float()
        obs = self.gym_env.reset()[0]
        agent_list = sorted(obs.keys())
        # Build a joint observation array (shape: [num_agents, obs_dim])
        obs = np.array([obs[agent] for agent in agent_list])
        state = obs.reshape(1, -1)  # to simulate global state in multi agent
        self.total_reward = 0
        info = {}
        return obs,state,self.action_mask,info


    def step(self,action,step_type=0):
        done_reward = None
        actions={agent: action[agent_idx] for agent_idx,agent in enumerate(self.gym_env.possible_agents)}
        # new_state, reward, is_done, is_tr, _ = self.gym_env.step(action[0].cpu().numpy())
        # if action[0].shape[0] == 1 and self.get_action_space().__class__.__name__ != 'Box':
        #     new_state, reward, is_done, is_tr, _ = self.gym_env.step(action[0][0])
        # else:
        #     new_state, reward, is_done, is_tr, _ = self.gym_env.step(action[0])
        next_obs, rewards, is_dones, is_trs, _=self.gym_env.step(actions)
        reward=rewards['walker_0']
        agent_list = sorted(next_obs.keys())
        # Build a joint observation array (shape: [num_agents, obs_dim])
        next_obs = np.array([next_obs[agent] for agent in agent_list])
        next_state = next_obs.reshape(1, -1)  # to simulate global state in multi agent
        is_done = is_dones['walker_0'] or is_trs['walker_0']
        self.total_reward += reward

        if is_done:
            done_reward = self.total_reward
            next_obs, next_state,_, _ = self.reset()
        # else:
        #     # next_obs = torch.tensor(new_state).to(self.device).float()
        #     next_obs = new_state
        #     next_state = next_obs.reshape(1, -1)  # to simulate global state in multi agent
        #     next_obs = np.repeat(next_state, self.num_tx_agents, axis=0)
        info = {'is_done': is_done, 'done_reward': done_reward}
        reward = np.array([[reward]])
        is_done = np.array([[is_done]])
        action_mask=self.action_mask
        return next_obs,next_state, action_mask,reward,is_done,info
