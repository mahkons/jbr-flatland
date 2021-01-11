import torch
import torch.nn as nn

class JudgeNetwork(nn.Module):
    def __init__(self, state_sz):
        super(JudgeNetwork, self).__init__()
        self.seq = nn.Sequential(
                nn.Linear(state_sz, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
        )

    # outputs logits of probabilities
    def forward(self, state):
        return self.seq(state)
