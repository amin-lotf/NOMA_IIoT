import numpy as np
import torch
from a0_config import all_configs
from ppo_joint.train_ppo_joint import main
add_to_seed=5
configs = all_configs.get_ppo_joint_config(add_to_seed)

if __name__ == '__main__':
    main(configs)
