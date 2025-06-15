from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.expert import MLPExpert
from layer.tower import MLPTower
from layer.attention import Attention, Info


class AITM(nn.Module):
    def __init__(self, config, feature_columns):
        super(AITM, self).__init__()
        conf = config['Model']
        self.num_experts = conf['num_experts']
        self.num_tasks = conf['num_tasks']
        self.units = conf['units']
        self.tower_out_units = conf['tower_units']
        self.tower_hidden_units = conf['tower_hidden_units']
        self.expert_hidden_units = conf['expert_hidden_units']
        self.dropout = conf['dropout']
        self.use_bn = conf['use_bn']

        self.condition_columns, self.sparse_feature_columns, self.dense_feature_columns = feature_columns
        self.condition_feature_columns_len = len(self.condition_columns)
        self.sparse_feature_columns_len = len(self.sparse_feature_columns)
        self.embedding = nn.ModuleList([nn.Embedding(feat['feat_onehot_dim'], feat['embed_dim']) for _, feat in
                                        enumerate(self.sparse_feature_columns)])

        self.num_feature = sum([feat['embed_dim'] for feat in self.sparse_feature_columns]) + len(
            self.dense_feature_columns)
        self.softmax = nn.Softmax(dim=1)

        self.expert = nn.ModuleList(
            [MLPExpert(self.num_feature, self.units, self.expert_hidden_units, self.dropout, self.use_bn) for _ in
             range(self.num_experts)])
        self.gate = nn.ModuleList([nn.Linear(self.num_feature, self.num_experts) for _ in range(self.num_tasks)])
        self.gate_softmax = nn.Softmax(dim=1)
        self.towers = nn.ModuleList([MLPTower(self.units + self.condition_feature_columns_len, self.tower_out_units,
                                              self.tower_hidden_units, self.dropout, self.use_bn) for _ in
                                     range(self.num_tasks)])
        self.infos = nn.ModuleList([Info(self.tower_out_units, self.dropout, self.use_bn) for _ in
                                    range(self.num_tasks - 1)])
        self.aits = nn.ModuleList([Attention(self.tower_out_units, bias=True) for _ in
                                    range(self.num_tasks - 1)])   # bias=True
        self.towers_head = nn.ModuleList([MLPTower(self.tower_out_units, 1, [], self.dropout, self.use_bn) for _ in
                                     range(self.num_tasks)])
        # self.bn = nn.BatchNorm1d(len(self.dense_feature_columns))
        self.flatten = nn.Flatten()

    def forward(self, x):
        condition_inputs, sparse_inputs, dense_inputs = x[:, :self.condition_feature_columns_len], x[:,self.condition_feature_columns_len:(self.condition_feature_columns_len + self.sparse_feature_columns_len)], x[:,(self.condition_feature_columns_len + self.sparse_feature_columns_len):]
        sparse_inputs = sparse_inputs.to(dtype=torch.long)
        sparse_embed = torch.cat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.embedding)],
                                 dim=1)  # [batch_size, 1, embed_dim]

        sparse_embed = self.flatten(sparse_embed)
        # Normalization
        # dense_inputs = self.bn(dense_inputs)
        inputs = torch.cat([sparse_embed, dense_inputs], dim=1)
        inputs = inputs.to(dtype=torch.float32)
        expert = [exp(inputs) for _, exp in enumerate(self.expert)]
        expert = torch.stack(expert, dim=1)  # [batch_size, num_experts, units]
        gate = [
            self.gate_softmax(g(inputs)).unsqueeze(2).repeat(1, 1, self.units) for _, g in
            enumerate(self.gate)]  # num_tasks * [batch_size, num_experts, units]

        expert = [torch.sum(g * expert, dim=1) for i, g in enumerate(gate)]  # num_tasks * [batch_size, units]
        tower = [t(torch.cat([expert[i], condition_inputs], dim=1)) for i, t in enumerate(self.towers)]

        info = [f(tower[i]) for i, f in enumerate(self.infos)]
        ait = [a(torch.cat([info[i].unsqueeze(1), tower[i+1].unsqueeze(1)], dim=1)) for i, a in enumerate(self.aits)]
        towers_head = [th(tower[i]) if i == 0 else th(ait[i-1]) for i, th in enumerate(self.towers_head)]

        out = torch.cat(towers_head, dim=1)  # [batch_size, num_tasks]
        return out

    def loss(self, out, label, weight=0.6):
        y_pred = torch.sigmoid(out)
        y_true = label
        loss_list = [F.binary_cross_entropy(y_pred[:, i], y_true[:, i]) for i in range(y_true.shape[-1])]

        constraint_loss = []
        for i in range(1, y_true.shape[-1]):
            label_constraint = torch.maximum(y_pred[:, i] - y_pred[:, i-1], torch.zeros_like(y_pred[:, i]))
            constraint_loss.append(torch.sum(label_constraint)/label_constraint.shape[0])
        
        loss = [loss_list[i] if i == 0 else loss_list[i] + weight * constraint_loss[i-1] for i in range(len(loss_list))]
        return (loss, loss_list, constraint_loss)
