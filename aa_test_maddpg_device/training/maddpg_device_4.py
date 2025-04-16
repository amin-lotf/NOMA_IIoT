import json

import numpy as np
import torch
from a0_config import all_configs
from maddpg_device.train_maddpg import main

add_to_seed=3
configs = all_configs.get_maddpg_device_config(add_to_seed)

if __name__ == '__main__':
    main(configs)
