import numpy as np
import torch
import torch.nn as nn

from .util import init, get_clones


"""MLP modules."""


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(fc_h, self._layer_N)

    def forward(self, x1):
        x = self.fc1(x1)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, input_dim):
        super(MLPBase, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_lstm = args.use_lstm
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)

        # If using LSTM, insert an LSTM layer after feature normalization.
        if self._use_lstm:
            # Here, we assume that x has shape [batch, seq, feature] and we use batch_first=True.
            # The LSTM takes input_dim as input size and outputs hidden_size features.
            self.lstm = nn.LSTM(input_dim, self.hidden_size, batch_first=True)
            mlp_input_dim = self.hidden_size
        else:
            mlp_input_dim = input_dim

        self.mlp = MLPLayer(mlp_input_dim, self.hidden_size,
                            self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        if self._use_lstm:
            # Ensure that the memory layout is contiguous
            x = x.contiguous()
            # Pass through LSTM; we ignore the hidden state output here.
            x, _ = self.lstm(x)
        x = self.mlp(x)
        return x


