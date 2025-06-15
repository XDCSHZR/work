from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.expert import MLPExpert
from layer.tower import MLPTower


class ESMM(nn.Module):
    def __init__(self, config, feature_columns):
        super(ESMM, self).__init__()
        # params
        # self.num_feature = config.num_feature
        conf = config.Model
        self.num_experts = conf['num_experts']
        self.num_tasks = 2
        self.units = conf['units']
        self.tower_hidden_units = conf['tower_hidden_units']
        self.expert_hidden_units = conf['expert_hidden_units']
        self.dropout = conf['dropout']
        self.use_bn = conf['use_bn']
        
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.sparse_feature_columns_len = len(self.sparse_feature_columns)
        self.embedding = nn.ModuleList([nn.Embedding(feat['feat_onehot_dim'], feat['embed_dim']) for _, feat in enumerate(self.sparse_feature_columns)])
        
        self.num_feature = sum([feat['embed_dim'] for feat in self.sparse_feature_columns]) + len(self.dense_feature_columns)
        self.softmax = nn.Softmax(dim=1)

        self.expert = nn.ModuleList([MLPExpert(self.num_feature, self.units, self.expert_hidden_units, self.dropout, self.use_bn) for _ in range(self.num_experts)])
        self.gate = nn.ModuleList([nn.Linear(self.num_feature, self.num_experts) for _ in range(self.num_tasks)])
        self.gate_softmax = nn.Softmax(dim=1)
        self.towers = nn.ModuleList([MLPTower(self.units, 1, self.tower_hidden_units, self.dropout, self.use_bn) for _ in range(self.num_tasks)])
        # print(len(self.dense_feature_columns))
        self.bn = nn.BatchNorm1d(len(self.dense_feature_columns))
        self.flatten = nn.Flatten()


    def forward(self, x):
        sparse_inputs, dense_inputs = x[:, :self.sparse_feature_columns_len], x[:, self.sparse_feature_columns_len:]
        sparse_inputs = sparse_inputs.to(dtype=torch.long)
        sparse_embed = torch.cat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.embedding)], dim=1)  # [batch_size, 1, embed_dim]
        # print('sparse_embed', sparse_embed.shape)
        sparse_embed = self.flatten(sparse_embed)
        # Normalization
        # print(dense_inputs.size())
        dense_inputs = self.bn(dense_inputs)
        inputs = torch.cat([sparse_embed, dense_inputs], dim=1)

        # print('inputs', inputs.shape)
        inputs = inputs.to(dtype=torch.float32)
        expert = [exp(inputs) for _, exp in enumerate(self.expert)]
        expert = torch.stack(expert, dim=1)  # [batch_size, num_experts, units]
        gate = [ 
            self.gate_softmax(g(inputs)).unsqueeze(2).repeat(1,1,self.units) for _, g in enumerate(self.gate)]  # num_tasks * [batch_size, num_experts, units]

        expert = [torch.sum(g * expert, dim=1) for i, g in enumerate(gate)]  # num_tasks * [batch_size, units]
        # tower = [t(e) for _, (t, e) in enumerate(zip(self.towers, expert))]
        tower = [t(expert[i]) for i, t in enumerate(self.towers)]
        tower = torch.cat(tower, dim=1)  # [batch_size, num_tasks]

        # out = self.softmax(tower)
        out = tower
        return out