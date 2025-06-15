import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.expert import MLPExpert, MLP
from layer.tower import MLPTower
from layer.attention import Attention, Info
from dataclasses import dataclass, field
from math import sqrt
from typing import List, Optional, Union


class AdaTTSpUnit(nn.Module):
    """
    paper title: "AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations"
    paper link: https://doi.org/10.1145/3580305.3599769
    Call Args:
        inputs: inputs is a tensor of dimension
            [batch_size, self.num_tasks, self.input_dim].
            Experts in the same module share the same input.
        outputs dimensions: [B, T, D_out]

    Example::
        AdaTTSp(
            input_dim=256,
            expert_out_dims=[[128, 128]],
            num_tasks=8,
            num_task_experts=2,
            self_exp_res_connect=True,
        )
    """

    def __init__(self, conf):
        super(AdaTTSpUnit, self).__init__()

        self.num_tasks = conf['num_tasks']
        self.num_task_experts = conf['num_experts'] # num_task_experts
        self.expert_out_dims = conf['output_units']  # expert_out_dims 
        self.total_experts_per_layer = self.num_task_experts * self.num_tasks
        if len(self.expert_out_dims) == 0:
            raise ValueError(
                "AdaTTSp is noop! size of expert_out_dims which is the number of extraction layers should be at least 1."
            )
        
        self.num_extraction_layers = len(self.expert_out_dims) # number of layers
        self.self_exp_res_connect = conf['exp_res_connect']
        self.input_dim = conf['input_units']  # input dim
        self.dropout = conf['dropout']
        self.use_bn = conf['use_bn']
        
        self.experts = torch.nn.ModuleList()
        self.gate_weights = torch.nn.ModuleList()

        self_exp_weight_list = []
        layer_input_dim = self.input_dim
        
        for expert_out_dim in self.expert_out_dims:
            self.experts.append(
                nn.ModuleList(
                    [
                        MLP(layer_input_dim, expert_out_dim, self.dropout, self.use_bn)
                        for i in range(self.total_experts_per_layer)
                    ]
                )
            )

            self.gate_weights.append(
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(
                                layer_input_dim, self.total_experts_per_layer
                            ),
                            nn.Softmax(dim=-1),
                        )
                        for _ in range(self.num_tasks)
                    ]
                )
            )  # self.gate_weights is of shape L X T, after we loop over all layers.

            if self.self_exp_res_connect and self.num_task_experts > 1:
                params = torch.empty(self.num_tasks, self.num_task_experts)
                scale = sqrt(1.0 / self.num_task_experts)
                torch.nn.init.uniform_(params, a=-scale, b=scale)
                self_exp_weight_list.append(torch.nn.Parameter(params))

            layer_input_dim = expert_out_dim[-1]

        # self.self_exp_weights = nn.ParameterList(self_exp_weight_list)
        # self.self_exp_weights = nn.ModuleList(self_exp_weight_list)
        self.self_exp_weights = self_exp_weight_list
        # print(len(self.self_exp_weights))
        
    def forward(self, inputs):
        # print('inputs', inputs.shape)
        inputs = inputs.to(dtype=torch.float32)    # [batch_size, feature_nums]
        
        inputs = inputs.unsqueeze(1).repeat(1, self.num_tasks, 1)

        # print(len(self.self_exp_weights), len(self.experts), len(self.gate_weights))
        for layer_i, (self_experts_layer_i, self_gate_weights_layer_i) in enumerate(zip(self.experts, self.gate_weights)):
            # all task expert outputs.
            experts_out = torch.stack(
                [
                    expert(inputs[:, expert_i // self.num_task_experts, :])
                    for expert_i, expert in enumerate(self_experts_layer_i)
                ],
                dim=1,
            )  # [B * E (total experts) * D_out]

            gates = torch.stack(
                [
                    gate_weight(
                        inputs[:, task_i, :]
                    )  #  W ([B, D]) * S ([D, E]) -> G, dim is [B, E]
                    for task_i, gate_weight in enumerate(self_gate_weights_layer_i)
                ],
                dim=1,
            )  # [B, T, E]
            fused_experts_out = torch.bmm(
                gates,
                experts_out,
            )  # [B, T, E] X [B * E (total experts) * D_out] -> [B, T, D_out]

            if self.self_exp_res_connect:
                if self.num_task_experts > 1:
                    # residual from the linear combination of tasks' own experts.
                    
                    # print(self.self_exp_weights[layer_i].to(experts_out.device).shape)
                    # print(self_exp_weights_layer_i.to(experts_out.device).shape)
                    self_exp_weighted = torch.einsum(
                        "te,bted->btd",
                        self.self_exp_weights[layer_i].to(experts_out.device),
                        # self_exp_weights_layer_i.to(experts_out.device),
                        experts_out.view(
                            experts_out.size(0),
                            self.num_tasks,
                            self.num_task_experts,
                            -1,
                        ),  # [B * E (total experts) * D_out] -> [B * T * E_task * D_out]
                    )  #  bmm: [T * E_task] X [B * T * E_task * D_out] -> [B, T, D_out]

                    fused_experts_out = (
                        fused_experts_out + self_exp_weighted
                    )  # [B, T, D_out]
                else:
                    fused_experts_out = fused_experts_out + experts_out

            inputs = fused_experts_out

        out = [inputs[:, i, :] for i in range(self.num_tasks)]
        return out


class AdaTTWSharedExpsUnit(nn.Module):
    """
    paper title: "AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations"
    paper link: https://doi.org/10.1145/3580305.3599769
    Call Args:
        inputs: inputs is a tensor of dimension
            [batch_size, self.num_tasks, self.input_dim].
            Experts in the same module share the same input.
        outputs dimensions: [B, T, D_out]

    Example::
        AdaTTWSharedExps(
            input_dim=256,
            expert_out_dims=[[128, 128]],
            num_tasks=8,
            num_shared_experts=1,
            num_task_experts=2,
            self_exp_res_connect=True,
        )
    """

    def __init__(self, conf):
        super(AdaTTWSharedExpsUnit, self).__init__()

        self.num_tasks = conf['num_tasks']
        self.num_task_experts = conf['num_task_experts']
        self.num_shared_experts = conf['num_shared_experts']
        self.expert_out_dims = conf['output_units']  # expert_out_dims
        # self.total_experts_per_layer = self.num_task_experts * self.num_tasks
        if len(self.expert_out_dims) == 0:
            raise ValueError(
                "AdaTTWSharedExps is noop! size of expert_out_dims which is the number of extraction layers should be at least 1."
            )
        
        self.num_extraction_layers = len(self.expert_out_dims) # number of layers
        self.self_exp_res_connect = conf['exp_res_connect']
        self.input_dim = conf['input_units']  # input_dim
        self.tower_hidden_units = conf['tower_hidden_units']
        self.dropout = conf['dropout']
        self.use_bn = conf['use_bn']
        self.tower_out_units = conf['tower_units']

        self.num_task_expert_list = conf['num_task_expert_list'] 
        print(self.num_task_expert_list)
        # Set num_task_expert_list for experimenting with a flexible number of
        # experts for different task_specific units.
        assert (self.num_task_experts is None) ^ (self.num_task_expert_list is None)
        if self.num_task_experts is not None:
            self.num_expert_list = [self.num_task_experts for _ in range(self.num_tasks)]
        else:
            # num_expert_list is guaranteed to be not None here.
            # pyre-ignore
            self.num_expert_list = self.num_task_expert_list
        self.num_expert_list.append(self.num_shared_experts)
        self.total_experts_per_layer = sum(self.num_expert_list)

        # self.input_layer = nn.Linear(self.num_feature, self.input_dim)
        self.experts = torch.nn.ModuleList()
        self.gate_weights = torch.nn.ModuleList()

        layer_input_dim = self.input_dim
        for layer_i, expert_out_dim in enumerate(self.expert_out_dims):
            self.experts.append(
                torch.nn.ModuleList(
                    [
                        MLP(layer_input_dim, expert_out_dim, self.dropout, self.use_bn)
                        for i in range(self.total_experts_per_layer)
                    ]
                )
            )

            num_full_active_modules = (
                self.num_tasks
                if layer_i == self.num_extraction_layers - 1
                else self.num_tasks + 1
            )

            self.gate_weights.append(
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(
                                layer_input_dim, self.total_experts_per_layer
                            ),
                            torch.nn.Softmax(dim=-1),
                        )
                        for _ in range(num_full_active_modules)
                    ]
                )
            )  # self.gate_weights is a 2d module list of shape L X T (+ 1), after we loop over all layers.

            layer_input_dim = expert_out_dim[-1]

        self_exp_weight_list = []
        if self.self_exp_res_connect:
            # If any tasks have number of experts not equal to 1, we learn linear combinations of native experts.
            if any(num_experts != 1 for num_experts in self.num_expert_list):
                for i in range(self.num_tasks + 1):
                    num_full_active_layer = (
                        self.num_extraction_layers - 1
                        if i == self.num_tasks
                        else self.num_extraction_layers
                    )
                    params = torch.empty(
                        num_full_active_layer,
                        self.num_expert_list[i],
                    )
                    scale = sqrt(1.0 / self.num_expert_list[i])
                    torch.nn.init.uniform_(params, a=-scale, b=scale)
                    self_exp_weight_list.append(torch.nn.Parameter(params))

        # self.self_exp_weights = nn.ParameterList(self_exp_weight_list)
        self.self_exp_weights = self_exp_weight_list
        self.expert_input_idx = []
        for i in range(self.num_tasks + 1):
            self.expert_input_idx.extend([i for _ in range(self.num_expert_list[i])])
        
        self.use_unit_naive_weights = all(num_expert == 1 for num_expert in self.num_expert_list)
        
    def forward(self, inputs):
        # print('inputs', inputs.shape)
        inputs = inputs.to(dtype=torch.float32)    # [batch_size, feature_nums]

        # inputs = self.input_layer(inputs)
        inputs = inputs.unsqueeze(1).repeat(1, self.num_tasks + 1, 1)

        # for layer_i in range(self.num_extraction_layers):
        # print(len(self.self_exp_weights), len(self.experts), len(self.gate_weights))
        for layer_i, (self_experts_layer_i, self_gate_weights_layer_i) in enumerate(zip(self.experts, self.gate_weights)):
            num_full_active_modules = (
                self.num_tasks
                if layer_i == self.num_extraction_layers - 1
                else self.num_tasks + 1
            )

            experts_out = torch.stack(
                [
                    expert(inputs[:, self.expert_input_idx[expert_i], :])
                    for expert_i, expert in enumerate(self_experts_layer_i)
                ],
                dim=1,
            )  # [B * E (total experts) * D_out]

            gates = torch.stack(
                [
                    gate_weight(
                        inputs[:, task_i, :]
                    )  #  W ([B, D]) * S ([D, E]) -> G, dim is [B, E]
                    for task_i, gate_weight in enumerate(self_gate_weights_layer_i)
                ],
                dim=1,
            )  # [B, T(+1), E]

            # add all expert gate weights with native expert weights.
            if self.self_exp_res_connect:
                prev_idx = 0
                for module_i in range(num_full_active_modules):
                    next_idx = self.num_expert_list[module_i] + prev_idx
                    if self.use_unit_naive_weights:
                        gates[:, module_i, prev_idx: next_idx] += torch.ones(1, self.num_expert_list[module_i]).to(gates.device)
                    else:
                        gates[:, module_i, prev_idx: next_idx] += self.self_exp_weights[module_i][layer_i].unsqueeze(0).to(gates.device)
                    prev_idx = next_idx

            fused_experts_out = torch.bmm(
                gates,
                experts_out,
            )  # [B, T(+1), E] X [B * E (total experts) * D_out] -> [B, T(+1), D_out]


            inputs = fused_experts_out

        out = [inputs[:, i, :] for i in range(self.num_tasks)]
        return out
    