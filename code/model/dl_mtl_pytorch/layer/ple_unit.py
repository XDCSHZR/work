"""
PLE and CGC unit network structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.expert import MLPExpert

# Customized gate control, single extraction layer
class CGCUnit(nn.Module):
    def __init__(self, specific_input, specific_output, shared_input, shared_output, is_last, conf):
        super(CGCUnit, self).__init__()
        self.num_tasks = conf['num_tasks']
        self.specific_hidden_units = conf['specific_hidden_units']
        self.shared_hidden_units = conf['shared_hidden_units']
        self.num_specific_expert = conf['num_specific_expert']
        self.num_shared_expert = conf['num_shared_expert']
        self.dropout = conf['dropout']
        self.use_bn = conf['use_bn']
        self.specific_input = specific_input
        self.shared_input = shared_input
        self.specific_output = specific_output
        self.shared_output = shared_output
        self.specific_expert = nn.ModuleList([nn.ModuleList(
            [MLPExpert(specific_input, specific_output, self.specific_hidden_units, self.dropout, self.use_bn) for _ in
             range(self.num_specific_expert)]) for _ in range(self.num_tasks)])  # num_tasks * specific expert
        self.shared_expert = nn.ModuleList(
            [MLPExpert(shared_input, shared_output, self.shared_hidden_units, self.dropout, self.use_bn) for _ in
             range(self.num_shared_expert)])  # 1 * shared expert
        self.is_last = is_last

        # if not last layer, use specific experts and shared expert
        if not self.is_last:
            n_specific = self.num_specific_expert + self.num_shared_expert
            n_shared = self.num_tasks * self.num_specific_expert + self.num_shared_expert
            self.gate = nn.ModuleList([nn.Linear(self.specific_input, n_specific) for _ in range(self.num_tasks)] + [
                nn.Linear(self.shared_input, n_shared)])
            self.gate_softmax = nn.ModuleList([nn.Softmax(dim=1) for _ in range(self.num_tasks + 1)])
        # if last layer, only use specific experts
        else:
            n_specific = self.num_specific_expert + self.num_shared_expert
            self.gate = nn.ModuleList([nn.Linear(self.specific_input, n_specific) for _ in range(self.num_tasks)])
            self.gate_softmax = nn.ModuleList([nn.Softmax(dim=1) for _ in range(self.num_tasks)])

        self.specific_shape = self.specific_output * self.num_specific_expert
        self.shared_shape = self.shared_output * self.num_shared_expert

    def forward(self, x, x_shared):  # 只考虑输出就行
        x = [x[:, :, i] for i in range(self.num_tasks)] + [x_shared]
        # x: [task0, task1, ..., taskn, shared]
        # gate = [self.gate_softmax[i](g(x[i])) for i, g in enumerate(self.gate)]  # gate[-1] is the shared gate
        gate = []

        for i, (sm, g) in enumerate(zip(self.gate_softmax, self.gate)):
            gate.append(sm(g(x[i])))
        # gate: [spec0_gate, spec1_gate, ..., specn_gate, shared_gate]

        # shared_expert_output = []  # [sh_expert0, sh_expert1, ...]
        # for _, expert in enumerate(self.shared_expert):
        #     shared_expert_output.append(expert(x[-1]))

        specific_expert_output = []  # [[spec0_expert0, spec0_expert1, ...], [spec1_expert0, spec1_expert1, ...], ...]

        for i, expert_list in enumerate(self.specific_expert):
            tmp_specific = []
            tmp_shared = []
            for j, expert in enumerate(expert_list):
                tmp_specific.append(expert(x[i]) * gate[i][:, j].unsqueeze(1).repeat(1, self.specific_output))

            for k, expert_sh in enumerate(self.shared_expert):
                tmp_shared.append(expert_sh(x[-1]) * gate[i][:, k + self.num_specific_expert].unsqueeze(1).repeat(1,
                                                                                                                  self.shared_output))

            # Flatten for concatenation
            # torch.stack(tmp_specific, dim=1).reshape(-1, self.specific_shape)
            # torch.stack(tmp_shared, dim=1).reshape(-1, self.shared_shape)
            specific_expert_output.append(torch.cat([torch.stack(tmp_specific, dim=1).reshape(-1, self.specific_shape),
                                                     torch.stack(tmp_shared, dim=1).reshape(-1, self.shared_shape)],
                                                    dim=1))

        if self.is_last:
            return torch.stack(specific_expert_output, dim=2), torch.tensor(0)
        else:
            shared_expert_output = []
            tmp_specific = []
            for i, expert_list1 in enumerate(self.specific_expert):
                for j, expert1 in enumerate(expert_list1):
                    tmp_specific.append(
                        expert1(x[i]) * gate[-1][:, i * self.num_specific_expert + j].unsqueeze(1).repeat(1,
                                                                                                          self.specific_output))

            tmp_shared = []
            for k, expert_sh1 in enumerate(self.shared_expert):
                tmp_shared.append(
                    expert_sh1(x[-1]) * gate[-1][:, self.num_tasks * self.num_specific_expert + k].unsqueeze(1).repeat(
                        1, self.shared_output))

            # Flatten for concatenation
            # torch.stack(tmp_specific, dim=1).reshape(-1, self.specific_shape * self.num_tasks)
            # torch.stack(tmp_shared, dim=1).reshape(-1, self.shared_shape)
            shared_expert_output.append(torch.cat(
                [torch.stack(tmp_specific, dim=1).reshape(-1, self.specific_shape * self.num_tasks),
                 torch.stack(tmp_shared, dim=1).reshape(-1, self.shared_shape)], dim=1))

            return torch.stack(specific_expert_output, dim=2), shared_expert_output[0]


class PLEUnit(nn.Module):
    def __init__(self, conf):
        super(PLEUnit, self).__init__()
        self.num_tasks = conf['num_tasks']
        self.specific_units = conf['specific_units']
        self.shared_units = conf['shared_units']
        self.num_specific_expert = conf['num_specific_expert']
        self.num_shared_expert = conf['num_shared_expert']
        self.num_extraction_module = conf['num_extraction_module']
        self.num_inputs = conf['num_inputs']
        self.dropout = conf['dropout']
        self.use_bn = conf['use_bn']
        assert self.num_extraction_module >= 1  # at least one extraction module

        if self.num_extraction_module == 1:
            self.extraction_module = nn.ModuleList([CGCUnit(self.num_inputs, self.specific_units, self.num_inputs,
                                                        self.shared_units, is_last=True, conf=conf)])
        elif self.num_extraction_module == 2:
            self.cgc_input = [
                CGCUnit(self.num_inputs, self.specific_units, self.num_inputs, self.shared_units, is_last=False,
                    conf=conf)]
            self.cgc_out = [
                CGCUnit(self.specific_units * self.num_specific_expert + self.shared_units * self.num_shared_expert,
                    self.specific_units,
                    self.specific_units * self.num_specific_expert * self.num_tasks + self.shared_units * self.num_shared_expert,
                    self.shared_units, is_last=True, conf=conf)]
            self.extraction_module = nn.ModuleList(self.cgc_input + self.cgc_out)
        else:
            self.cgc_input = [
                CGCUnit(self.num_inputs, self.specific_units, self.num_inputs, self.shared_units, is_last=False,
                    conf=conf)]
            self.cgc_mid = [CGCUnit(
                self.specific_units * self.num_specific_expert + self.shared_units * self.num_shared_expert,
                self.specific_units,
                self.specific_units * self.num_specific_expert * self.num_tasks + self.shared_units * self.num_shared_expert,
                self.shared_units, is_last=False, conf=conf)] * (self.num_extraction_module - 2)
            self.cgc_out = [
                CGCUnit(self.specific_units * self.num_specific_expert + self.shared_units * self.num_shared_expert,
                    self.specific_units,
                    self.specific_units * self.num_specific_expert * self.num_tasks + self.shared_units * self.num_shared_expert,
                    self.shared_units, is_last=True, conf=conf)]
            self.extraction_module = nn.ModuleList(self.cgc_input + self.cgc_mid + self.cgc_out)


    def forward(self, inputs):
        inputs = inputs.to(dtype=torch.float32)

        # inputs -> [task0, task1, ..., taskn, shared]
        inputs_specific = torch.stack([inputs] * self.num_tasks, dim=2)
        inputs_shared = inputs

        for _, extraction in enumerate(self.extraction_module):
            inputs_specific, inputs_shared = extraction(inputs_specific, inputs_shared)

        inputs_specific = [inputs_specific[:, :, i] for i in range(self.num_tasks)]
        # tower = [t(inputs_specific[i]) for i, t in enumerate(self.towers)]
        # out = torch.cat(tower, dim=1)  # [batch_size, num_tasks]
        return inputs_specific

