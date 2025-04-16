import numpy as np
import torch
from a0_config import all_configs
from pso.train_pso import main

add_to_seed=6
configs = all_configs.get_pso_config(add_to_seed,is_testing=True)

if __name__ == '__main__':
    main(configs)
