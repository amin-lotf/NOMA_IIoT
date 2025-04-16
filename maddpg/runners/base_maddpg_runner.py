from pathlib import Path
import numpy as np
import torch
from general_utils.gens import load_pretrained_actor_critic
from maddpg.algorithms.policy import VectorizedAttentionMADDPGPolicy
from maddpg.algorithms.trainer import VectorizedAttentionMADDPGTrainer
from general_utils import gens
from maddpg.maddpg_utils.maddpg_priority_buffer import MADDPGPriorityReplayBuffer


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
        self.maddpg_offloading_config = self.env.maddpg_offloading_config
        self.maddpg_tx_config = self.env.maddpg_tx_config
        if self.testing_env:
            self.maddpg_offloading_config.epsilon_start=0.0
            self.maddpg_offloading_config.epsilon_end=0.0
            self.maddpg_tx_config.epsilon_start = 0.0
            self.maddpg_tx_config.epsilon_end = 0.0
        # self.writer = SummaryWriter(comment='maddpg-offload')
        self.device = self.env.device
        self.algorithm_name = self.maddpg_offloading_config.algorithm_name
        self.num_offloading_agents = self.env.num_offloading_agents
        self.num_tx_agents = self.env.num_tx_agents
        self.num_envs = self.maddpg_offloading_config.n_envs
        self.num_env_steps = self.maddpg_tx_config.num_env_steps
        self.offloading_save_interval = self.maddpg_offloading_config.save_interval
        self.tx_save_interval = self.maddpg_tx_config.save_interval
        self.use_offloading_linear_lr_decay = self.maddpg_offloading_config.use_linear_lr_decay
        self.use_tx_linear_lr_decay = self.maddpg_tx_config.use_linear_lr_decay
        self.env.metadata['is_single_agent'] = self.maddpg_tx_config.single_agent

        self.env.metadata['n_sbs'] = self.env.n_sbs
        self.env.metadata['max_users'] = self.env.max_users

        # ===== Start Offloading ===== #
        self.env.metadata['act_type'] = 'offloading'
        self.env.metadata['split_quantization'] = self.maddpg_offloading_config.split_quantization
        self.env.metadata['num_agents'] = self.env.num_offloading_agents
        self.offloading_policy = VectorizedAttentionMADDPGPolicy(self.maddpg_offloading_config,
                                                                 self.env.num_offloading_agents,
                                                                 self.env.get_observation_space(1),
                                                                 self.env.get_state_space(1),
                                                                 self.env.get_action_space(1),
                                                                 device=self.device,
                                                                 metadata=self.env.metadata)
        if self.env.maddpg_offloading_config.load_pretrained_weights:
            load_pretrained_actor_critic(self.env, self.offloading_policy, self.maddpg_offloading_config,
                                         is_offloading=True)
        self.offloading_trainer = VectorizedAttentionMADDPGTrainer(self.maddpg_offloading_config, self.offloading_policy,
                                                device=self.device)
        self.offloading_buffer = MADDPGPriorityReplayBuffer(self.maddpg_offloading_config,
                                                    self.env.num_offloading_agents,
                                                    self.env.get_observation_space(1),
                                                    self.env.get_state_space(1),
                                                    self.env.get_action_space(1),
                                                    self.env.get_available_action_shape(1),
                                                    self.device,
                                                    shift_reward=self.env.max_task_deadline)
        # ===== End Offloading ===== #
        # ===== Start NOMA ===== #
        self.env.metadata['act_type'] = 'noma'
        self.env.metadata['max_users'] = self.env.max_users
        self.env.metadata['num_agents'] = self.env.num_tx_agents
        self.tx_policy = VectorizedAttentionMADDPGPolicy(self.maddpg_tx_config,
                                                         self.env.num_tx_agents,
                                                         self.env.get_observation_space(0),
                                                         self.env.get_state_space(0),
                                                         self.env.get_action_space(0),
                                                         device=self.device,
                                                         metadata=self.env.metadata)
        if self.env.maddpg_tx_config.load_pretrained_weights:
            load_pretrained_actor_critic(self.env, self.tx_policy, self.maddpg_tx_config, is_offloading=False)
        self.tx_trainer = VectorizedAttentionMADDPGTrainer(self.maddpg_tx_config, self.tx_policy, device=self.device)
        self.tx_buffer = MADDPGPriorityReplayBuffer(self.maddpg_tx_config,
                                            self.env.num_tx_agents,
                                            self.env.get_observation_space(0),
                                            self.env.get_state_space(0),
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

    def collect(self, obs,available_actions,is_offloading=False):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data, is_offloading=False):
        raise NotImplementedError


    def train(self, is_offloading=False):
        """Train policies with data in offloading_buffer. """
        trainer = self.offloading_trainer if is_offloading else self.tx_trainer
        buffer = self.offloading_buffer if is_offloading else self.tx_buffer
        trainer.prep_training()
        if self.testing_env:
            #Remove this if else
            train_infos={}
            # train_infos = trainer.train(buffer)
        else:
            train_infos = trainer.train(buffer)
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
        torch.save(policy.critic.state_dict(), f'{path_str}/{env_str}{env_name}_critic.pt')

    def unmerge(self, values, num_agents):
        return values.reshape((self.num_envs, num_agents, *values.shape[1:]))

    def merge(self, values):
        return values.reshape((-1, *values.shape[2:]))
