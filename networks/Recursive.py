import torch
import torch.nn as nn
import torch.nn.functional as F

GENERAL_SZ = 9
FEATURES_SZ = 14
HIDDEN_SZ = 64

def _get_depth(sz):
    for i in range(10):
        if 2 ** i == sz + 2:
            return i - 1
    assert False

class RecursiveLayer(nn.Module):
    def __init__(self, state_sz, out_sz):
        super(RecursiveLayer, self).__init__()
        self.encode_features = nn.Sequential(nn.Linear(FEATURES_SZ, HIDDEN_SZ), nn.ReLU(inplace=True))
        self.merge_value = nn.Sequential(
                nn.Linear(3 * HIDDEN_SZ, 3 * HIDDEN_SZ),
                nn.ReLU(inplace=True),
                nn.Linear(3 * HIDDEN_SZ, HIDDEN_SZ),
                nn.ReLU(inplace=True)
            )

        self.encode_general = nn.Sequential(nn.Linear(GENERAL_SZ, 2 * HIDDEN_SZ), nn.ReLU(inplace=True),
                nn.Linear(2 * HIDDEN_SZ, HIDDEN_SZ), nn.ReLU(inplace=True))
        self.final = nn.Sequential(nn.Linear(2 * HIDDEN_SZ + HIDDEN_SZ, 128), nn.ReLU(inplace=True), nn.Linear(128, out_sz))

    def forward(self, state):
        b_sz = state.shape[0]
        general = state[:, :GENERAL_SZ]
        features = self.encode_features(state[:, GENERAL_SZ:].view(b_sz, -1, FEATURES_SZ))
        # features shape (bsz, 2**(depth+1) - 2, HIDDEN_SZ)

        depth = _get_depth(features.shape[1])
        merged = [torch.zeros((b_sz, 2 ** (lvl + 1), HIDDEN_SZ), device=features.device) for lvl in range(depth + 1)]

        for d in range(depth, 0, -1):
            l, r = 2 ** d - 2, 2 ** (d + 1) - 2
            lc, rc = 2 ** (d + 1) - 2, 2 ** (d + 2) - 2
            cur_features = features[:, l:r]
            children_features = merged[d].view(b_sz, 2**d, 2 * HIDDEN_SZ)
            merged[d - 1] = self.merge_value(torch.cat([cur_features, children_features], dim=2))

        general = self.encode_general(general)
        #  values = self.merge_children(merged[0].view(b_sz, 2 * HIDDEN_SZ)) # merge or not to merge
        values = merged[0].view(b_sz, 2 * HIDDEN_SZ)

        return self.final(torch.cat([general, values], dim=1))
