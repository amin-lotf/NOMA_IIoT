import numpy as np
import torch
from a0_config import all_configs
from duel_ddqn.train_duel_ddqn import main
add_to_seed=2
configs = all_configs.get_duel_ddqn_config(add_to_seed,is_testing=True)

if __name__ == '__main__':
    main(configs)
