import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.feature_norm = nn.LayerNorm(input_dim)
        self.apply(init_weights)

    def forward(self, x):
        x = self.feature_norm(x)
        residual = x
        out = self.fc1(x)
        out = F.relu(self.ln1(out))
        out = self.fc2(out)
        out = self.ln2(out)
        out += residual
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks):
        super(ResNet, self).__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(input_dim) for _ in range(num_blocks)])
        self.final_fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.blocks(x)
        x = self.final_fc(x)
        return x
