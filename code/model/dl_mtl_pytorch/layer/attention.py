from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, bias=True):  # Q,K,V bias is True as default
        super(Attention, self).__init__()
        self.dim = dim
        self.q_layer = nn.Linear(dim, dim, bias=bias)
        self.k_layer = nn.Linear(dim, dim, bias=bias)
        self.v_layer = nn.Linear(dim, dim, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        # nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, inputs):
        Q = self.q_layer(inputs)
        K = self.k_layer(inputs)
        V = self.v_layer(inputs)
        a = torch.sum(torch.mul(Q, K), -1) / torch.sqrt(torch.tensor(self.dim).to(torch.double))
        a = self.softmax(a)
        outputs = torch.sum(torch.mul(torch.unsqueeze(a, -1), V), dim=1)
        return outputs


class Info(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5, use_bn=False):
        super(Info, self).__init__()
        net = []
        net.append(("info_fc", nn.Linear(input_size, input_size)))
        if use_bn:
            net.append(("info_bn", nn.BatchNorm1d(input_size)))
        net.append(("info_relu", nn.ReLU()))
        net.append(("info_dropout", nn.Dropout(dropout_rate)))

        self.info = nn.Sequential(OrderedDict(net))

    def forward(self, x):
        return self.info(x)
