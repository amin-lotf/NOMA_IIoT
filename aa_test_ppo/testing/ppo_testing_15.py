import numpy as np
import torch
from a0_config import all_configs
from ppo.train_ppo import main
add_to_seed=14
configs = all_configs.get_ppo_config(add_to_seed,is_testing=True)

if __name__ == '__main__':
    main(configs)
