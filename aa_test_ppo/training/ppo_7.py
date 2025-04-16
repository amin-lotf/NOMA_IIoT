import numpy as np
import torch
from a0_config import all_configs
from ppo.train_ppo import main
add_to_seed=6
configs = all_configs.get_ppo_config(add_to_seed)

if __name__ == '__main__':
    main(configs)
