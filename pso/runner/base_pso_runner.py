


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
        self.pgoa_offloading_config = self.env.pgoa_offloading_config
        self.pgoa_tx_config = self.env.pgoa_tx_config
        # self.writer = SummaryWriter(comment='pgoa-offload')
        self.device = self.env.device
        self.algorithm_name = self.pgoa_offloading_config.algorithm_name
        self.num_envs = self.pgoa_offloading_config.n_envs
        self.num_env_steps = self.pgoa_offloading_config.num_env_steps
        self.offloading_save_interval = self.pgoa_offloading_config.save_interval
        self.tx_save_interval = self.pgoa_tx_config.save_interval


    def run(self, logging_config):
        """Collect training data, perform training updates, and evaluate offloading_policy."""
        raise NotImplementedError

    def warmup(self, is_offloading=False):
        """Collect warmup pre-training data."""
        raise NotImplementedError


