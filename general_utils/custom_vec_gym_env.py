from typing import Type, List

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices


class CustomVecGymEnv(VecEnv):
    """
    A custom vectorized environment that:
      - Works with gymnasium (i.e. reset returns (obs, info), step returns (obs, reward, terminated, truncated, info))
      - Returns an extra "action_mask" along with observations on both reset and step.

    It assumes that all environments have the same observation and action spaces.

    Parameters
    ----------
    envs: list
        A list of gymnasium environments.
    """

    def __init__(self, envs):
        self.envs = envs
        self.num_envs = len(self.envs)

        # Assume all envs share the same observation and action spaces.
        observation_space = self.envs[0].get_observation_space(0)
        self.state_space = self.envs[0].get_state_space(0)
        action_space = self.envs[0].get_action_space(0)

        # Call the parent constructor.
        super().__init__(num_envs=self.num_envs,
                         observation_space=observation_space,
                         action_space=action_space)

    def _aggregate_info(self, infos,step_type=0):
        """
        Given a list of info dictionaries from all environments, compute the elementwise mean
        for the keys 'noma_metrics' and 'fl_metrics' (if they are not None).

        If a metric is already a NumPy array for any environment, it will be kept as an array
        in the averaged result.

        Returns
        -------
        aggregated_info: dict
            A dictionary with keys 'noma_metrics' and 'fl_metrics' where each value is the
            elementwise mean computed across the corresponding arrays from all environments,
            or None if no environment provided data for that key.
        """
        aggregated_info = {}
        for key in ['done_reward']:
            values = []
            for info in infos:
                val = info.get(key)
                if val is not None:
                    # If it's already a NumPy array, use it as is; otherwise, convert it.
                    if isinstance(val, np.ndarray):
                        values.append(val)
                    else:
                        values.append(np.array(val))
            if values:
                # Stack the arrays along a new axis and compute the mean along that axis.
                aggregated_info[key] = np.mean(np.stack(values, axis=0), axis=0)
            else:
                aggregated_info[key] = None
        return aggregated_info

    def get_post_data(self,step_type=0):
        observations = []
        states = []
        action_masks = []
        for env in self.envs:
            obs, state, action_mask = env.get_post_data(step_type=step_type)
            observations.append(obs)
            states.append(state)
            action_masks.append(action_mask)
        observations = np.array(observations)
        states = np.array(states)
        action_masks = np.array(action_masks) if action_masks[0] is not None else None
        return observations, states, action_masks

    def reset(self, seed=None, options=None,step_type=0):
        """
        Reset all environments.

        Returns
        -------
        observations : np.ndarray
            The observations returned by each env.
        action_masks : np.ndarray or None
            The corresponding action masks (if available).
        info : dict
            A single dictionary containing aggregated metrics:
                - 'noma_metrics': elementwise mean across all envs (or None)
                - 'fl_metrics': elementwise mean across all envs (or None)
        """
        observations = []
        states = []
        action_masks = []
        infos = []
        for env in self.envs:
            obs,state, action_mask, info = env.reset(seed=seed, options=options,step_type=step_type)
            observations.append(obs)
            states.append(state)
            action_masks.append(action_mask)
            infos.append(info)

        observations = np.array(observations)
        states = np.array(states)
        action_masks = np.array(action_masks) if action_masks[0] is not None else None

        # Return a single aggregated info dictionary.
        aggregated_info = self._aggregate_info(infos,step_type)
        return observations,states, action_masks, aggregated_info

    def step_async(self, actions):
        """
        Begin stepping the environments with the provided actions.
        The actions are stored to be used in step_wait.
        """
        self.actions = actions

    def step_wait(self,step_type=0):
        """
        Step all environments with the actions provided in the previous call to step_async.

        Returns
        -------
        observations : np.ndarray
            Observations after the step.
        action_masks : np.ndarray or None
            The corresponding action masks (if available).
        rewards : np.ndarray
            Rewards received after the step.
        terminateds : np.ndarray
            Flags indicating termination.
        info : dict
            A single dictionary containing aggregated metrics:
                - 'noma_metrics': elementwise mean across all envs (or None)
                - 'fl_metrics': elementwise mean across all envs (or None)
        """
        observations = []
        states = []
        action_masks = []
        rewards = []
        terminateds = []
        infos = []

        for i, env in enumerate(self.envs):
            # gymnasium step returns 5 values.
            obs, state, action_mask, reward, terminated, info = env.step(self.actions[i],step_type=step_type)
            observations.append(obs)
            states.append(state)
            action_masks.append(action_mask)
            rewards.append(reward)
            terminateds.append(terminated)
            infos.append(info)

        observations = np.array(observations)
        states = np.array(states)
        action_masks = np.array(action_masks) if action_masks[0] is not None else None
        rewards = np.array(rewards)
        terminateds = np.array(terminateds)

        # Return a single aggregated info dictionary.
        aggregated_info = self._aggregate_info(infos,step_type=step_type)
        return observations, states, action_masks, rewards, terminateds, aggregated_info

    def step(self, actions,step_type=0):
        """
        A synchronous step: calls step_async then step_wait.
        """
        self.step_async(actions)
        return self.step_wait(step_type)

    def close(self):
        """
        Close all environments.
        """
        for env in self.envs:
            env.close()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call a method on each of the environments (or a subset).
        """
        results = []
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            results.append(getattr(self.envs[i], method_name)(*method_args, **method_kwargs))
        return results

    def get_attr(self, attr_name, indices=None):
        """
        Get attribute from all environments (or a subset).
        """
        results = []
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            results.append(getattr(self.envs[i], attr_name))
        return results

    def set_attr(self, attr_name, value, indices=None):
        """
        Set an attribute on all environments (or a subset).
        """
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            setattr(self.envs[i], attr_name, value)

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        # Implementation left as an exercise.
        pass
