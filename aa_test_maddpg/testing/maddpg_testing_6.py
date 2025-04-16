import json

import numpy as np
import torch
from a0_config import all_configs
from maddpg.train_maddpg import main

add_to_seed=5
configs = all_configs.get_maddpg_config(add_to_seed,is_testing=True)

if __name__ == '__main__':
    main(configs)
