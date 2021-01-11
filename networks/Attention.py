import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, state_sz, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.scale = 1.0 / state_sz ** 0.5

        self.fck = nn.Linear(state_sz, state_sz * n_head, bias=False)
        self.fcq = nn.Linear(state_sz, state_sz * n_head, bias=False)
        self.fcv = nn.Linear(state_sz, state_sz * n_head, bias=False)

    # key shape (batch_size, keys_len, state_sz)
    # query shape (batch_size, queries_len, state_sz)
    def forward(self, query, key, value):
        batch_sz, len_k, len_q, len_v, state_sz = key.shape[0], key.shape[1], query.shape[1], value.shape[1], key.shape[2]
        assert len_k == len_v


        key = self.fck(key).view(batch_sz, len_k, self.n_head, state_sz)
        query = self.fcq(query).view(batch_sz, len_q, self.n_head, state_sz)
        value = self.fcv(value).view(batch_sz, len_v, self.n_head, state_sz)

        key, query, value = key.transpose(1, 2), query.transpose(1, 2), value.transpose(1, 2)
        weights = F.softmax(torch.matmul(query, key.transpose(2, 3)) * self.scale, dim=-1)
        # weights shape (batch_sz, n_head, query_len, key_len)

        output = torch.matmul(weights, value)
        # output shape (batch_sz, n_head, query_len, state_sz)

        # do we need a residual connection with batch_norm? 
        # shrug makes a lot of sense when there are several attention layers, yet we have only one

        # out shape (batch_sz, query_len, state_sz * n_head)
        return torch.flatten(output.transpose(1, 2), start_dim=2)
        

        

