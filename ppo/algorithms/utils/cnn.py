import torch.nn as nn
from .util import init

import torch
"""CNN Modules and utils."""

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNLayer(nn.Module):
    def __init__(self, obs_shape, args, use_orthogonal, use_ReLU, kernel_size=3, stride=1):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=args.hidden_size_cnn ,
                            kernel_size=8,
                            stride=4)
                  ),
            active_func,
            init_(nn.Conv2d(in_channels=args.hidden_size_cnn,
                            out_channels=args.hidden_size_cnn*2,
                            kernel_size=4,
                            stride=2)
                  ),
            active_func,
            init_(nn.Conv2d(in_channels=args.hidden_size_cnn*2,
                            out_channels=args.hidden_size_cnn*2,
                            kernel_size=3,
                            stride=1)
                  ),
            active_func,
            Flatten())
        size = self.cnn(torch.zeros(1, *obs_shape)).size()[-1]
        self.fc=nn.Sequential(nn.Linear(size,
                            args.hidden_size),
            active_func,
            nn.Linear(args.hidden_size, args.hidden_size), active_func)

        # self.fc = nn.Sequential(NoisyLinear(size,
        #                                     args.hidden_size),
        #                         active_func)

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        x=self.fc(x)
        return x

    # def reset_noise(self):
    #     # pass
    #     self.fc[0].reset_noise()
    #     # self.fc[2].reset_noise()


class CNNBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size

        self.cnn = CNNLayer(obs_shape, args, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        x = self.cnn(x)
        return x

    # def reset_noise(self):
    #     self.cnn.reset_noise()