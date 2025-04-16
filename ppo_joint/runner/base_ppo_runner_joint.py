from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from general_utils.gens import load_pretrained_actor_critic
from ppo.ppo_env import PPOEnv
from ppo.ppo_utils.shared_buffer import SharedReplayBuffer
from general_utils import gens
from ppo.algorithms.ppo_trainer import PPOTrainer
from ppo.algorithms.ppo_policy import PPOPolicy


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """

    def __init__(self, envs):
        self.env = envs.envs[0]
        self.envs = envs
        self.testing_env = self.env.env_config.testing_env
        self.ppo_config = self.env.ppo_config
        # self.writer = SummaryWriter(comment='PPO-offload')
        self.device = self.env.device
        self.algorithm_name = self.ppo_config.algorithm_name
        self.num_agents = self.env.num_agents
        self.num_envs = self.ppo_config.n_envs
        self.num_env_steps = self.ppo_config.num_env_steps
        self.episode_length = self.ppo_config.episode_length + self.env.max_task_deadline
        self.save_interval = self.ppo_config.save_interval
        self.use_linear_lr_decay = self.ppo_config.use_linear_lr_decay
        self.env.metadata['is_single_agent'] = self.ppo_config.single_agent

        self.env.metadata['n_sbs'] = self.env.n_sbs
        self.env.metadata['max_users'] = self.env.max_users
        # ===== Start Joint ===== #
        self.env.metadata['act_type'] = 'joint'
        self.env.metadata['split_quantization'] = self.ppo_config.split_quantization
        self.env.metadata['num_agents'] = self.num_agents
        self.policy = PPOPolicy(self.ppo_config,
                                   self.env.get_observation_space(2),
                                   self.env.get_state_space(2),
                                   self.env.get_action_space(2),
                                   device=self.device,
                                   metadata=self.env.metadata)
        if self.env.ppo_config.load_pretrained_weights:
            load_pretrained_actor_critic(self.env, self.policy, self.ppo_config, is_offloading=True)
        self.trainer = PPOTrainer(self.ppo_config, self.policy, device=self.device)
        self.buffer = SharedReplayBuffer(self.ppo_config,
                                            self.num_agents,
                                            self.env.get_observation_space(2),
                                            self.env.get_state_space(2),
                                            self.env.get_action_space(2),
                                            self.env.get_available_action_shape(2),
                                            self.device,
                                            shift_reward=self.env.max_task_deadline)

    def run(self, logging_config):
        """Collect training data, perform training updates, and evaluate offloading_policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        trainer = self.trainer
        buffer = self.buffer
        n_agents = self.num_agents
        trainer.prep_rollout()
        next_values = trainer.policy.get_values(self.merge(buffer.state[-1]),
                                                self.merge(buffer.rnn_states_critic[-1]),
                                                self.merge(buffer.masks[-1]))
        next_values = self.unmerge(next_values, n_agents)
        buffer.compute_returns(next_values, trainer.value_normalizer)

    def train(self):
        """Train policies with data in offloading_buffer. """
        trainer =  self.trainer
        buffer = self.buffer
        trainer.prep_training()
        # if is_offloading==False:
        #     train_infos={}
        #     # train_infos = trainer.train(buffer)
        # else:
        #     train_infos = {}
        #     # train_infos = trainer.train(buffer)
        if self.testing_env:
            train_infos={}
        else:
            train_infos = trainer.train(buffer)
        buffer.after_update()
        return train_infos

    def save_model(self):
        env_name = self.env.env_name
        glob_env_name = self.env.glob_env_name
        env_str = 'offloading_'
        path_str = f'{gens.get_project_root()}/saved_models/{glob_env_name}'
        policy =  self.policy
        if not Path(path_str).is_dir():
            Path(path_str).mkdir(parents=True, exist_ok=True)
        torch.save(policy.actor.state_dict(), f'{path_str}/{env_str}{env_name}_actor.pt')
        torch.save(policy.critic.state_dict(), f'{path_str}/{env_str}{env_name}_critic.pt')

    def unmerge(self, values, num_agents):
        return values.reshape((self.num_envs, num_agents, *values.shape[1:]))

    def merge(self, values):
        return values.reshape((-1, *values.shape[2:]))
