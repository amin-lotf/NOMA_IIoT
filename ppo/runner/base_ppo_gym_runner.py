from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from fl_utils.fed import fed_avg
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
        self.ppo_tx_config = self.env.ppo_tx_config
        self.writer = SummaryWriter(comment='MAPPO')
        self.device = self.env.device
        self.algorithm_name = self.ppo_tx_config.algorithm_name
        self.num_tx_agents = self.env.num_tx_agents
        self.num_envs = self.ppo_tx_config.n_envs
        self.num_env_steps = self.ppo_tx_config.num_env_steps
        self.tx_episode_length = self.ppo_tx_config.episode_length
        self.tx_save_interval = self.ppo_tx_config.save_interval
        self.use_tx_linear_lr_decay = self.ppo_tx_config.use_linear_lr_decay

        self.tx_policy = PPOPolicy(self.ppo_tx_config,
                                   self.env.get_observation_space(0),
                                   self.env.get_state_space(0),
                                   self.env.get_action_space(0),
                                   device=self.device)
        if self.env.ppo_tx_config.load_pretrained_weights:
            load_pretrained_actor_critic(self.env, self.tx_policy, self.ppo_tx_config, is_offloading=False)
        self.tx_trainer = PPOTrainer(self.ppo_tx_config, self.tx_policy, device=self.device)
        self.tx_buffer = SharedReplayBuffer(self.ppo_tx_config,
                                            self.env.num_tx_agents,
                                            self.env.get_observation_space(0),
                                            self.env.get_state_space(0),
                                            self.env.get_action_space(0),
                                            self.env.get_available_action_shape(0),
                                            self.device)
        # ===== End NOMA ===== #

    def run(self):
        """Collect training data, perform training updates, and evaluate offloading_policy."""
        raise NotImplementedError

    def warmup(self, is_offloading=False):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step, is_offloading=False):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data, is_offloading=False):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self, is_offloading=False):
        """Calculate returns for the collected data."""
        trainer =  self.tx_trainer
        buffer =  self.tx_buffer
        n_agents =  self.num_tx_agents
        trainer.prep_rollout()
        next_values = trainer.policy.get_values(self.merge(buffer.state[-1]),
                                                self.merge(buffer.rnn_states_critic[-1]),
                                                self.merge(buffer.masks[-1]))
        next_values = self.unmerge(next_values, n_agents)
        buffer.compute_returns(next_values, trainer.value_normalizer)

    def train(self, is_offloading=False):
        """Train policies with data in offloading_buffer. """
        trainer =  self.tx_trainer
        buffer =  self.tx_buffer
        trainer.prep_training()
        # if is_offloading==False:
        #     train_infos={}
        #     # train_infos = trainer.train(buffer)
        # else:
        #     train_infos = {}
        #     # train_infos = trainer.train(buffer)
        train_infos = trainer.train(buffer)
        buffer.after_update()
        return train_infos


    def unmerge(self, values, num_agents):
        return values.reshape((self.num_envs, num_agents, *values.shape[1:]))

    def merge(self, values):
        return values.reshape((-1, *values.shape[2:]))
