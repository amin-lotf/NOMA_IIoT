import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def compute_target_entropy(act_spaces):
    target_entropy = 0.0
    # Iterate over each action space in the list
    for space in act_spaces:
        if hasattr(space, 'shape'):  # Typically for Box spaces
            # Using negative of product of dimensions
            target_entropy += -np.prod(space.shape)
        elif hasattr(space, 'n'):  # Typically for Discrete spaces
            # A simple heuristic: use -log(probability of a uniform action)
            target_entropy += -np.log(1.0 / space.n)
        elif hasattr(space, 'nvec'):  # Typically for MultiDiscrete spaces
            # Sum the entropies for each discrete dimension
            target_entropy += -np.sum(np.log(1.0 / np.array(space.nvec)))
        else:
            raise ValueError("Unsupported action space type")
    return target_entropy
