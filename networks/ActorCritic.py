import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.Attention import MultiHeadAttention
from networks.Recursive import RecursiveLayer

from env.observations.SimpleObservation import ObservationDecoder

def create_linear_layers(in_sz, out_sz, layers_sz):
    layers = list()
    for sz in layers_sz:
        layers += [nn.Linear(in_sz, sz), nn.ReLU(inplace=True)]
        in_sz = sz
    layers.append(nn.Linear(in_sz, out_sz))
    return layers

THOUGHT_SZ = 128
INTENTION_SZ = 32
N_HEAD = 5

MAX_NEIGHBOURS_DEPTH = 5
_direction_tensor = [None]
for n_d in range(1, MAX_NEIGHBOURS_DEPTH + 1):
    dt = torch.zeros((2 ** (n_d + 1) - 2, n_d), dtype=torch.float)
    for _d in range(1, n_d + 1):
        lp = 2 ** (_d - 1) - 2
        rp = 2 ** _d - 2
        l = 2 ** _d - 2
        r = 2 ** (_d + 1) - 2
        if _d > 1:
            dt[l:r:2] = dt[lp:rp]
            dt[l+1:r:2] = dt[lp:rp]
        dt[l:r:2, _d - 1] = -1
        dt[l+1:r:2, _d - 1] = 1
    _direction_tensor.append(dt)


def _add_direction(signals, neighbours_depth):
    direction_tensor = _direction_tensor[neighbours_depth].to(signals.device)
    return torch.cat([signals, direction_tensor.unsqueeze(0).expand(signals.shape[0], -1, -1)], dim=2)


class ActorNet(nn.Module):
    def __init__(self, state_sz, action_sz, layers_sz, neighbours_depth):
        super(ActorNet, self).__init__()

        self.neighbours_depth = neighbours_depth
        self.think_seq = nn.Sequential(RecursiveLayer(state_sz, THOUGHT_SZ))

        self.fc_thought = nn.Linear(THOUGHT_SZ, INTENTION_SZ + self.neighbours_depth)
        self.signal_attention = MultiHeadAttention(state_sz=INTENTION_SZ + self.neighbours_depth, n_head=N_HEAD)
        self.signal_seq = nn.Sequential(nn.Linear(THOUGHT_SZ, INTENTION_SZ, bias=False), nn.ReLU(inplace=True))

        assert action_sz == 3
        self.first_head = nn.Sequential(nn.Linear(THOUGHT_SZ + (INTENTION_SZ + self.neighbours_depth) * N_HEAD, 256),
                nn.ReLU(inplace=True), nn.Linear(256, 3))
        self.second_head = nn.Sequential(nn.Linear(THOUGHT_SZ + (INTENTION_SZ + self.neighbours_depth) * N_HEAD, 256),
                nn.ReLU(inplace=True), nn.Linear(256, 2))

    def think(self, state):
        return self.think_seq(state)

    def intent(self, thought):
        return self.signal_seq(thought)

    # thought_shape = (batch_size, THOUGHT_SZ)
    # signals_shape = (batch_size, AGENT_NUMBER, INTENTION_SZ)
    def act(self, states, thought, signals):
        query = self.fc_thought(thought).unsqueeze(1)
        key = _add_direction(signals, self.neighbours_depth)
        attended_signals = self.signal_attention(query=query, key=key, value=key).squeeze(1)

        input = torch.cat([thought, attended_signals], dim=1)

        mask = torch.tensor([ObservationDecoder.is_real(obs, 1) for obs in states], dtype=torch.bool, device=thought.device)
        
        # extremely ugly code yeah
        first_log = self.first_head(input[mask])
        second_log = self.second_head(input[~mask])

        bsz = states.shape[0]
        second_log = torch.cat([-torch.ones((bsz - mask.sum(), 1), device=thought.device) * 1e6, second_log], dim=1)

        actions = torch.empty((bsz, 3), device=thought.device)
        actions[mask] = first_log
        actions[~mask] = second_log
        return actions


class CriticNet(nn.Module):
    def __init__(self, state_sz, action_sz, layers_sz):
        super(CriticNet, self).__init__()
        layers = create_linear_layers(state_sz, 1, layers_sz)
        self.seq = nn.Sequential(RecursiveLayer(state_sz, 1))

    def forward(self, state):
        return self.seq(state)
