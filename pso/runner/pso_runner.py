from general_utils import OffloadingPerformanceTracker, TxPerformanceTracker
from pso.runner.base_pso_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class AlgoRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, env):
        super(AlgoRunner, self).__init__(env)

    def run(self, logging_config):
        slot_per_seconds=1/self.env.slot_duration
        glob_step=0
        tx_episode=0
        offloading_episode=0
        tx_step=0
        offloading_step=0
        logging_config.save_period=200
        with (OffloadingPerformanceTracker(self.env, logging_config) as offloading_tracker):
            with TxPerformanceTracker(self.env, logging_config,fast_results=True) as tx_tracker:
                self.warmup(is_offloading=False)
                while glob_step <= self.num_env_steps:
                    if   glob_step%slot_per_seconds==0:
                        rewards, info = self.envs.step(step_type=0)
                        # tx_reward = tx_rewards[0]
                        noma_metrics = info['noma_metrics']
                        if noma_metrics is not None:
                            tx_tracker.record_performance(noma_metrics, rewards[0])
                        tx_step+=1
                        if tx_step % self.tx_save_interval == 0:
                            print(
                               f" Time slot: {self.env.cur_timeslot}, TX reward: {rewards[0].item()}")

                    rewards,  info = self.envs.step( step_type=1)
                    # tx_reward = tx_rewards[0]
                    offloading_metrics = info['offloading_metrics']
                    if offloading_metrics is not None:
                        offloading_tracker.record_performance(offloading_metrics, rewards[0])

                    offloading_step += 1
                    if offloading_step % self.offloading_save_interval == 0:
                        print(
                            f" Time slot: {self.env.cur_timeslot}, offloading reward: {rewards[0].item()}")
                    glob_step+=1
                print('End of the simulation!')

    def warmup(self,is_offloading=False):
        step_type=  is_offloading
        self.envs.reset(step_type=step_type)


