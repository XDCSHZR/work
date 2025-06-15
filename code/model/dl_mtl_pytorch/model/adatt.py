import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.expert import MLPExpert, MLP
from layer.tower import MLPTower
from layer.attention import Attention, Info
from dataclasses import dataclass, field
from math import sqrt
from typing import List, Optional, Union


class AdaTTSp(nn.Module):
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

    def __init__(self, config, feature_columns):
        super(AdaTTSp, self).__init__()

        conf = config.Model
        self.num_tasks = conf['num_tasks']
        self.num_task_experts = conf['num_task_experts']
        self.expert_out_dims = conf['expert_out_dims']
        self.total_experts_per_layer = self.num_task_experts * self.num_tasks
        if len(self.expert_out_dims) == 0:
            raise ValueError(
                "AdaTTSp is noop! size of expert_out_dims which is the number of extraction layers should be at least 1."
            )
        
        self.num_extraction_layers = len(self.expert_out_dims) # number of layers
        self.self_exp_res_connect = conf['exp_res_connect']
        self.input_dim = conf['input_dim']
        self.tower_hidden_units = conf['tower_hidden_units']
        self.dropout = conf['dropout']
        self.use_bn = conf['use_bn']
        self.tower_out_units = conf['tower_units']

        self.dense_feature_columns, self.sparse_feature_columns, self.condition_columns = feature_columns
        self.sparse_feature_columns_len = len(self.sparse_feature_columns)
        self.condition_feature_columns_len = len(self.condition_columns)
        self.embedding = nn.ModuleList([nn.Embedding(feat['feat_onehot_dim'], feat['embed_dim']) for _, feat in enumerate(self.sparse_feature_columns)])

        self.num_feature = sum([feat['embed_dim'] for feat in self.sparse_feature_columns]) + len(
            self.dense_feature_columns)

        # self.input_layer = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Linear(self.num_feature, self.input_dim), 
        #             nn.ReLU(),
        #         ) for _ in range(self.total_experts_per_layer)
        #     ]
        # )
        
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
        self.self_exp_weights = self_exp_weight_list
        
        self.towers = nn.ModuleList([MLPTower(self.expert_out_dims[-1][-1] + self.condition_feature_columns_len, 1, self.tower_hidden_units, self.dropout, self.use_bn) for _ in range(self.num_tasks)])
        self.bn = nn.BatchNorm1d(len(self.dense_feature_columns))
        self.flatten = nn.Flatten()
        
        self.towers = nn.ModuleList([MLPTower(self.expert_out_dims[-1][-1] + self.condition_feature_columns_len, self.tower_out_units, self.tower_hidden_units, self.dropout, self.use_bn) for _ in
                                     range(self.num_tasks)])
        self.infos = nn.ModuleList([Info(self.tower_out_units, self.dropout, self.use_bn) for _ in
                                    range(self.num_tasks - 1)])
        self.aits = nn.ModuleList([Attention(self.tower_out_units, bias=True) for _ in
                                    range(self.num_tasks - 1)])   # bias=True
        self.towers_head = nn.ModuleList([MLPTower(self.tower_out_units, 1, [], self.dropout, self.use_bn) for _ in
                                     range(self.num_tasks)])
        
    def forward(self, x):
        condition_inputs, sparse_inputs, dense_inputs = x[:, :self.condition_feature_columns_len], x[:,self.condition_feature_columns_len:(self.condition_feature_columns_len + self.sparse_feature_columns_len)], x[:,(self.condition_feature_columns_len + self.sparse_feature_columns_len):]
        sparse_inputs = sparse_inputs.to(dtype=torch.long)
        sparse_embed = torch.cat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.embedding)],
                                 dim=1)  # [batch_size, 1, embed_dim]
        # print('sparse_embed', sparse_embed.shape)
        sparse_embed = self.flatten(sparse_embed)
        # Normalization
        # print(dense_inputs.size())
        dense_inputs = self.bn(dense_inputs)
        inputs = torch.cat([sparse_embed, dense_inputs], dim=1)

        # print('inputs', inputs.shape)
        inputs = inputs.to(dtype=torch.float32)    # [batch_size, feature_nums]
        # inputs = self.input_layer(inputs)
        
        inputs = inputs.unsqueeze(1).repeat(1, self.num_tasks, 1)

        # for layer_i in range(self.num_extraction_layers):
        # print(len(self.self_exp_weights), len(self.experts), len(self.gate_weights))
        for layer_i, (self_experts_layer_i, self_gate_weights_layer_i) in enumerate(zip(self.experts, self.gate_weights)):
            # all task expert outputs.
            # self_experts_layer_i = self.experts[layer_i]
            # self_gate_weights_i = self.gate_weights[layer_i]
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

        tower = [t(torch.cat([inputs[:, i, :], condition_inputs], dim=1)) for i, t in enumerate(self.towers)]
        info = [f(tower[i]) for i, f in enumerate(self.infos)]
        ait = [a(torch.cat([info[i].unsqueeze(1), tower[i+1].unsqueeze(1)], dim=1)) for i, a in enumerate(self.aits)]
        # print(ait[0].shape)
        towers_head = [th(tower[i]) if i == 0 else th(ait[i-1]) for i, th in enumerate(self.towers_head)]

        out = torch.cat(towers_head, dim=1)  # [batch_size, num_tasks]
        return out
    
    def loss(self, out, label, x, constraint_weight=0.6, product_weight=[1.1, 3.5, 3, 0.5, 0.5, 0.5, 2.5, 2.5]):
        y_pred = torch.sigmoid(out)
        y_true = label
        condition_inputs = x[:, :self.condition_feature_columns_len]
        
        condition_shape = condition_inputs.shape[1]
        # auto complete
        if len(product_weight) >= condition_shape:
            product_weight = product_weight[:condition_shape]
        else:
            product_weight = product_weight + [1] * (condition_shape - len(product_weight))
            
        product_weight = torch.tensor([product_weight], device=y_pred.device)
        product_weight = product_weight.repeat(condition_inputs.shape[0], 1) * condition_inputs
        product_weight = product_weight.sum(axis=1)
        
        loss_list = [torch.mean(F.binary_cross_entropy(y_pred[:, i], y_true[:, i], reduction='none') * product_weight, axis=0) for i in range(y_true.shape[-1])]
        # loss_list = [F.binary_cross_entropy(y_pred[:, i], y_true[:, i]) for i in range(y_true.shape[-1])]

        constraint_loss = []
        for i in range(1, y_true.shape[-1]):
            label_constraint = torch.maximum(y_pred[:, i] - y_pred[:, i-1], torch.zeros_like(y_pred[:, i]))
            constraint_loss.append(torch.sum(label_constraint * product_weight)/label_constraint.shape[0])
        
        # loss = sum(loss_list) + constraint_weight * sum(constraint_loss)
        loss = [loss_list[i] if i == 0 else loss_list[i] + constraint_weight * constraint_loss[i-1] for i in range(len(loss_list))]
        return (loss, loss_list, constraint_loss)



class AdaTTWSharedExps(nn.Module):
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

    def __init__(self, config, feature_columns):
        super(AdaTTWSharedExps, self).__init__()

        conf = config.Model
        self.num_tasks = conf['num_tasks']
        self.num_task_experts = conf['num_task_experts']
        self.num_shared_experts = conf['num_shared_experts']
        self.expert_out_dims = conf['expert_out_dims']
        # self.total_experts_per_layer = self.num_task_experts * self.num_tasks
        if len(self.expert_out_dims) == 0:
            raise ValueError(
                "AdaTTWSharedExps is noop! size of expert_out_dims which is the number of extraction layers should be at least 1."
            )
        
        self.num_extraction_layers = len(self.expert_out_dims) # number of layers
        self.self_exp_res_connect = conf['exp_res_connect']
        self.input_dim = conf['input_dim']
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

        self.dense_feature_columns, self.sparse_feature_columns, self.condition_columns = feature_columns
        self.sparse_feature_columns_len = len(self.sparse_feature_columns)
        self.condition_feature_columns_len = len(self.condition_columns)
        self.embedding = nn.ModuleList([nn.Embedding(feat['feat_onehot_dim'], feat['embed_dim']) for _, feat in enumerate(self.sparse_feature_columns)])

        self.num_feature = sum([feat['embed_dim'] for feat in self.sparse_feature_columns]) + len(
            self.dense_feature_columns)

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

        self.towers = nn.ModuleList([MLPTower(self.expert_out_dims[-1][-1] + self.condition_feature_columns_len, 1, self.tower_hidden_units, self.dropout, self.use_bn) for _ in range(self.num_tasks)])
        self.bn = nn.BatchNorm1d(len(self.dense_feature_columns))
        self.flatten = nn.Flatten()
        
        self.towers = nn.ModuleList([MLPTower(self.expert_out_dims[-1][-1] + self.condition_feature_columns_len, self.tower_out_units, self.tower_hidden_units, self.dropout, self.use_bn) for _ in
                                     range(self.num_tasks)])
        self.infos = nn.ModuleList([Info(self.tower_out_units, self.dropout, self.use_bn) for _ in
                                    range(self.num_tasks - 1)])
        self.aits = nn.ModuleList([Attention(self.tower_out_units, bias=True) for _ in 
                                   range(self.num_tasks - 1)])   # bias=True
        self.towers_head = nn.ModuleList([MLPTower(self.tower_out_units, 1, [], self.dropout, self.use_bn) for _ in
                                     range(self.num_tasks)])
        
        self.use_unit_naive_weights = all(num_expert == 1 for num_expert in self.num_expert_list)
        
    def forward(self, x):
        condition_inputs, sparse_inputs, dense_inputs = x[:, :self.condition_feature_columns_len], x[:,self.condition_feature_columns_len:(self.condition_feature_columns_len + self.sparse_feature_columns_len)], x[:,(self.condition_feature_columns_len + self.sparse_feature_columns_len):]
        sparse_inputs = sparse_inputs.to(dtype=torch.long)
        sparse_embed = torch.cat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.embedding)],
                                 dim=1)  # [batch_size, 1, embed_dim]
        # print('sparse_embed', sparse_embed.shape)
        sparse_embed = self.flatten(sparse_embed)
        # Normalization
        # print(dense_inputs.size())
        dense_inputs = self.bn(dense_inputs)
        inputs = torch.cat([sparse_embed, dense_inputs], dim=1)

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

        tower = [t(torch.cat([inputs[:, i, :], condition_inputs], dim=1)) for i, t in enumerate(self.towers)]
        info = [f(tower[i]) for i, f in enumerate(self.infos)]
        ait = [a(torch.cat([info[i].unsqueeze(1), tower[i+1].unsqueeze(1)], dim=1)) for i, a in enumerate(self.aits)]
        # print(ait[0].shape)
        towers_head = [th(tower[i]) if i == 0 else th(ait[i-1]) for i, th in enumerate(self.towers_head)]

        out = torch.cat(towers_head, dim=1)  # [batch_size, num_tasks]
        return out
    
    def loss(self, out, label, x, constraint_weight=0.6, product_weight=[1.5, 4, 4, 0.5, 0.5, 0.5, 3.5, 4]):
        y_pred = torch.sigmoid(out)
        y_true = label
        condition_inputs = x[:, :self.condition_feature_columns_len]
        
        condition_shape = condition_inputs.shape[1]
        # auto complete
        if len(product_weight) >= condition_shape:
            product_weight = product_weight[:condition_shape]
        else:
            product_weight = product_weight + [1] * (condition_shape - len(product_weight))
            
        product_weight = torch.tensor([product_weight], device=y_pred.device)
        product_weight = product_weight.repeat(condition_inputs.shape[0], 1) * condition_inputs
        product_weight = product_weight.sum(axis=1)
        
        loss_list = [torch.mean(F.binary_cross_entropy(y_pred[:, i], y_true[:, i], reduction='none') * product_weight, axis=0) for i in range(y_true.shape[-1])]
        # loss_list = [F.binary_cross_entropy(y_pred[:, i], y_true[:, i]) for i in range(y_true.shape[-1])]

        constraint_loss = []
        for i in range(1, y_true.shape[-1]):
            label_constraint = torch.maximum(y_pred[:, i] - y_pred[:, i-1], torch.zeros_like(y_pred[:, i]))
            constraint_loss.append(torch.sum(label_constraint * product_weight)/label_constraint.shape[0])
        
        # loss = sum(loss_list) + constraint_weight * sum(constraint_loss)
        loss = [loss_list[i] if i == 0 else loss_list[i] + constraint_weight * constraint_loss[i-1] for i in range(len(loss_list))]
        return (loss, loss_list, constraint_loss)
    
#     def loss(self, out, label, weight=0.6):
#         y_pred = torch.sigmoid(out)
#         y_true = label
#         loss_list = [F.binary_cross_entropy(y_pred[:, i], y_true[:, i]) for i in range(y_true.shape[-1])]

#         constraint_loss = []
#         for i in range(1, y_true.shape[-1]):
#             label_constraint = torch.maximum(y_pred[:, i] - y_pred[:, i-1], torch.zeros_like(y_pred[:, i]))
#             constraint_loss.append(torch.sum(label_constraint)/label_constraint.shape[0])
        
#         # loss = sum(loss_list) + weight * sum(constraint_loss)
#         loss = [loss_list[i] if i == 0 else loss_list[i] + weight * constraint_loss[i-1] for i in range(len(loss_list))]
#         return (loss, loss_list, constraint_loss)

