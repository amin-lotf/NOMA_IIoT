from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from duel_ddqn.duel_ddqn_algorithms.r_madqn.algorithm.duel_ddqn_policy import DuelDDQNPolicy
from duel_ddqn.duel_ddqn_algorithms.r_madqn.duel_ddqn_trainer import MADQNTrainer
from duel_ddqn.duel_ddqn_utils.duel_ddqn_priority_buffer import DQNPriorityBuffer
from general_utils.gens import load_pretrained_actor_critic, load_pretrained_value_based_model

from general_utils import gens



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
        self.duel_ddqn_offloading_config = self.env.duel_ddqn_offloading_config
        self.duel_ddqn_tx_config = self.env.duel_ddqn_tx_config
        if self.testing_env:
            self.duel_ddqn_offloading_config.dqn_epsilon_start=0.0
            self.duel_ddqn_offloading_config.dqn_epsilon_end=0.0
            self.duel_ddqn_tx_config.dqn_epsilon_start = 0.0
            self.duel_ddqn_tx_config.dqn_epsilon_end = 0.0

        self.device = self.env.device
        self.writer = SummaryWriter(comment='duel_ddqn')
        self.algorithm_name = self.duel_ddqn_offloading_config.algorithm_name
        self.num_offloading_agents = self.env.num_offloading_agents
        self.num_tx_agents = self.env.num_tx_agents
        self.num_envs = self.duel_ddqn_offloading_config.n_envs
        self.num_env_steps = self.duel_ddqn_offloading_config.num_env_steps
        self.offloading_save_interval = self.duel_ddqn_offloading_config.save_interval
        self.tx_save_interval = self.duel_ddqn_tx_config.save_interval
        self.use_offloading_linear_lr_decay = self.duel_ddqn_offloading_config.use_linear_lr_decay
        self.use_tx_linear_lr_decay = self.duel_ddqn_tx_config.use_linear_lr_decay
        self.env.metadata['is_single_agent'] = self.duel_ddqn_tx_config.single_agent

        self.env.metadata['n_sbs'] = self.env.n_sbs
        self.env.metadata['max_users'] = self.env.max_users

        # ===== Start Offloading ===== #
        self.env.metadata['act_type'] = 'offloading'
        self.env.metadata['split_quantization'] = self.duel_ddqn_offloading_config.split_quantization
        self.env.metadata['num_agents'] = self.env.num_offloading_agents
        self.offloading_policy = DuelDDQNPolicy(self.duel_ddqn_offloading_config,
                                           self.env.get_observation_space(1),
                                           self.env.get_action_space(1),
                                           device=self.device,
                                           metadata=self.env.metadata)
        self.offloading_target_policy = DuelDDQNPolicy(self.duel_ddqn_offloading_config,
                                                self.env.get_observation_space(1),
                                                self.env.get_action_space(1),
                                                device=self.device,
                                                metadata=self.env.metadata)
        if self.env.duel_ddqn_offloading_config.load_pretrained_weights:
            load_pretrained_value_based_model(self.env, self.offloading_policy, self.offloading_target_policy,
                                              self.duel_ddqn_tx_config, is_offloading=True)
        self.offloading_trainer = MADQNTrainer(self.duel_ddqn_tx_config, self.offloading_policy,self.offloading_target_policy, device=self.device)
        self.offloading_buffer = DQNPriorityBuffer(self.duel_ddqn_offloading_config,
                                                    self.env.num_offloading_agents,
                                                    self.env.get_observation_space(1),
                                                    self.env.get_action_space(1),
                                                    self.env.get_available_action_shape(1),
                                                    self.device,
                                                    shift_reward=self.env.max_task_deadline)
        # ===== End Offloading ===== #
        # ===== Start NOMA ===== #
        self.env.metadata['act_type'] = 'noma'
        self.env.metadata['max_users'] = self.env.max_users
        self.env.metadata['num_agents'] = self.env.num_tx_agents
        self.tx_policy = DuelDDQNPolicy(self.duel_ddqn_tx_config,
                                        self.env.get_observation_space(0),
                                        self.env.get_action_space(0),
                                        device=self.device,
                                        metadata=self.env.metadata)
        self.tx_target_policy = DuelDDQNPolicy(self.duel_ddqn_tx_config,
                                               self.env.get_observation_space(0),
                                               self.env.get_action_space(0),
                                               device=self.device,
                                               metadata=self.env.metadata)
        if self.env.duel_ddqn_tx_config.load_pretrained_weights:
            load_pretrained_value_based_model(self.env, self.tx_policy, self.tx_target_policy,
                                              self.duel_ddqn_tx_config, is_offloading=False)
        self.tx_trainer = MADQNTrainer(self.duel_ddqn_tx_config, self.tx_policy, self.tx_target_policy,
                                       device=self.device)
        self.tx_buffer = DQNPriorityBuffer(self.duel_ddqn_tx_config,
                                   self.env.num_tx_agents,
                                   self.env.get_observation_space(0),
                                   self.env.get_action_space(0),
                                   self.env.get_available_action_shape(0),
                                   self.device)
        # ===== End NOMA ===== #

    def run(self, logging_config):
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



    def train(self, is_offloading=False):
        """Train policies with data in offloading_buffer. """
        trainer = self.offloading_trainer if is_offloading else self.tx_trainer
        buffer = self.offloading_buffer if is_offloading else self.tx_buffer
        if not self.testing_env:
            trainer.prep_training()
            train_infos = trainer.train(buffer)
        else:
            train_infos={}
        # trainer.prep_training()
        # train_infos = trainer.train(buffer)
        return train_infos

    def save_model(self, is_offloading=False):
        env_name = self.env.env_name
        glob_env_name = self.env.glob_env_name
        env_str = 'offloading_' if is_offloading else 'noma_'
        path_str = f'{gens.get_project_root()}/saved_models/{glob_env_name}'
        policy = self.offloading_policy if is_offloading else self.tx_policy
        if not Path(path_str).is_dir():
            Path(path_str).mkdir(parents=True, exist_ok=True)
        torch.save(policy.actor.state_dict(), f'{path_str}/{env_str}{env_name}_actor.pt')

    def unmerge(self, values, num_agents):
        return values.reshape((self.num_envs, num_agents, *values.shape[1:]))

    def merge(self, values):
        return values.reshape((-1, *values.shape[2:]))
