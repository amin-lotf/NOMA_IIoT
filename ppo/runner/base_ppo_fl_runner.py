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
        self.ppo_offloading_config = self.env.ppo_offloading_config
        self.ppo_tx_config = self.env.ppo_tx_config
        if self.testing_env:
            self.ppo_offloading_config.entropy_coef=0.0
            self.ppo_offloading_config.entropy_coef=0.0
        self.writer = SummaryWriter(comment='PPO-offload')
        self.joint_decision = self.env.joint_decision
        self.device = self.env.device
        self.algorithm_name = self.ppo_offloading_config.algorithm_name
        self.num_offloading_agents = self.env.num_offloading_agents
        self.num_tx_agents = self.env.num_tx_agents
        self.num_envs = self.ppo_offloading_config.n_envs
        self.num_env_steps = self.ppo_tx_config.num_env_steps
        self.offloading_episode_length = self.ppo_offloading_config.episode_length + self.env.max_task_deadline
        self.tx_episode_length = self.ppo_tx_config.episode_length + (
            self.env.max_task_deadline if self.joint_decision else 0)
        self.offloading_save_interval = self.ppo_offloading_config.save_interval
        self.tx_save_interval = self.ppo_tx_config.save_interval
        self.use_offloading_linear_lr_decay = self.ppo_offloading_config.use_linear_lr_decay
        self.use_tx_linear_lr_decay = self.ppo_tx_config.use_linear_lr_decay
        self.use_tx_decay_entropy = self.ppo_tx_config.use_decay_entropy
        self.use_offloading_decay_entropy = self.ppo_offloading_config.use_decay_entropy
        self.env.metadata['is_single_agent'] = self.ppo_tx_config.single_agent

        self.env.metadata['n_sbs'] = self.env.n_sbs
        self.env.metadata['max_users'] = self.env.max_users
        if self.joint_decision:
            # ===== Start Joint ===== #
            self.env.metadata['act_type'] = 'joint'
            self.env.metadata['split_quantization'] = self.ppo_offloading_config.split_quantization
            self.env.metadata['num_agents'] = self.env.num_tx_agents
            self.tx_policy = PPOPolicy(self.ppo_tx_config,
                                       self.env.get_observation_space(2),
                                       self.env.get_state_space(2),
                                       self.env.get_action_space(2),
                                       device=self.device,
                                       metadata=self.env.metadata)
            if self.env.ppo_tx_config.load_pretrained_weights:
                load_pretrained_actor_critic(self.env, self.tx_policy, self.ppo_tx_config, is_offloading=False)
            self.tx_trainer = PPOTrainer(self.ppo_tx_config, self.tx_policy, device=self.device)
            self.tx_buffer = SharedReplayBuffer(self.ppo_tx_config,
                                                self.env.num_tx_agents,
                                                self.env.get_observation_space(2),
                                                self.env.get_state_space(2),
                                                self.env.get_action_space(2),
                                                self.env.get_available_action_shape(2),
                                                self.device,
                                                shift_reward=self.env.max_task_deadline)
            # ===== End Joint ===== #
        else:
            # ===== Start Offloading ===== #
            self.env.metadata['act_type'] = 'offloading'
            self.env.metadata['split_quantization'] = self.ppo_offloading_config.split_quantization
            self.env.metadata['num_agents'] = self.env.num_offloading_agents
            self.offloading_policy = PPOPolicy(self.ppo_offloading_config,
                                               self.env.get_observation_space(1),
                                               self.env.get_state_space(1),
                                               self.env.get_action_space(1),
                                               device=self.device,
                                               metadata=self.env.metadata)
            if self.env.ppo_offloading_config.load_pretrained_weights:
                load_pretrained_actor_critic(self.env, self.offloading_policy, self.ppo_offloading_config,
                                             is_offloading=True)
            self.offloading_trainer = PPOTrainer(self.ppo_offloading_config, self.offloading_policy, device=self.device)
            self.offloading_buffer = SharedReplayBuffer(self.ppo_offloading_config,
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
            self.tx_policy = PPOPolicy(self.ppo_tx_config,
                                       self.env.get_observation_space(0),
                                       self.env.get_state_space(0),
                                       self.env.get_action_space(0),
                                       device=self.device,
                                       metadata=self.env.metadata)
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

    @torch.no_grad()
    def compute(self, is_offloading=False):
        """Calculate returns for the collected data."""
        trainer = self.offloading_trainer if is_offloading else self.tx_trainer
        buffer = self.offloading_buffer if is_offloading else self.tx_buffer
        n_agents = self.num_offloading_agents if is_offloading else self.num_tx_agents
        trainer.prep_rollout()
        next_values = trainer.policy.get_values(self.merge(buffer.state[-1]),
                                                self.merge(buffer.rnn_states_critic[-1]),
                                                self.merge(buffer.masks[-1]))
        next_values = self.unmerge(next_values, n_agents)
        buffer.compute_returns(next_values, trainer.value_normalizer)

    def train(self, is_offloading=False):
        """Train policies with data in offloading_buffer. """
        trainer = self.offloading_trainer if is_offloading else self.tx_trainer
        buffer = self.offloading_buffer if is_offloading else self.tx_buffer
        trainer.prep_training()
        if is_offloading==True:
            if self.ppo_offloading_config.load_pretrained_weights:
                train_infos={}
            else:
                train_infos = trainer.train(buffer)
        else:
            if self.ppo_tx_config.load_pretrained_weights:
                train_infos = {}
            else:
                train_infos = trainer.train(buffer)
        # train_infos = trainer.train(buffer)
        buffer.after_update()
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
