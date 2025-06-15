"""
Tower network structure.
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


# poso
class GateNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, expert_num, dropout):
        super().__init__()
        layers = list()
        layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        layers.append(torch.nn.ReLU())
        for i in range(len(hidden_dims)-1):
            layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(torch.nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
        layers.append(torch.nn.Linear(hidden_dims[-1], expert_num)) #hidden_dims[-1]))
        layers.append(torch.nn.Sigmoid())
        self.gate_net = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, input_dim)``
        """
        return self.gate_net(x)