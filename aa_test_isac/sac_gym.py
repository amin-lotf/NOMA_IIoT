import numpy as np
import torch
from a0_config import all_configs
from isac.train_sac_gym import main

add_to_seed=0
configs = all_configs.get_isac_config(add_to_seed)

if __name__ == '__main__':
    main(configs)

