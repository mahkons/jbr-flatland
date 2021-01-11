import torch
import torch.nn as nn


class DQNNet(nn.Module):
    def __init__(self, state_sz, action_sz, layers_sz, device):
        super(DQNNet, self).__init__()
        self.layers_sz = layers_sz
        self.device = device
    
        layers = self.create_linear_layers(state_sz, action_sz, layers_sz)

        self.seq = nn.Sequential(*layers)

    def create_linear_layers(self, state_sz, action_sz, layers_sz):
        layers = list()
        in_sz = state_sz
        for sz in layers_sz:
            layers += [nn.Linear(in_sz, sz), nn.ReLU(inplace=True)]
            in_sz = sz
        layers.append(nn.Linear(in_sz, action_sz))
        return layers

    def forward(self, x):
        return self.seq(x)
