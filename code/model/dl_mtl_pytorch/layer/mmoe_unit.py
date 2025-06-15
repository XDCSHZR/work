"""
MMOE unit network structure.
"""
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.expert import MLPExpert


class MMOEUnit(nn.Module):
    def __init__(self, conf):
        super(MMOEUnit, self).__init__()
        # params
        # self.num_feature = config.num_feature
        self.num_experts = conf['num_experts']
        self.num_tasks = conf['num_tasks']
        self.output_units = conf['output_units']
        self.dropout = conf['dropout']
        self.use_bn = conf['use_bn']
        self.input_units = conf['input_units']
        self.expert_hidden_units = conf['expert_hidden_units']

        self.softmax = nn.Softmax(dim=1)

        self.expert = nn.ModuleList(
            [MLPExpert(self.input_units, self.output_units, self.expert_hidden_units, self.dropout, self.use_bn) for _ in
             range(self.num_experts)])
        self.gate = nn.ModuleList([nn.Linear(self.input_units, self.num_experts) for _ in range(self.num_tasks)])
        self.gate_softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # print('inputs', inputs.shape)
        inputs = inputs.to(dtype=torch.float32)
        expert = [exp(inputs) for _, exp in enumerate(self.expert)]
        expert = torch.stack(expert, dim=1)  # [batch_size, num_experts, output_units]
        gate = [
            self.gate_softmax(g(inputs)).unsqueeze(2).repeat(1, 1, self.output_units) for _, g in
            enumerate(self.gate)]  # num_tasks * [batch_size, num_experts, output_units]

        expert = [torch.sum(g * expert, dim=1) for i, g in enumerate(gate)]  # num_tasks * [batch_size, output_units]
        return expert
