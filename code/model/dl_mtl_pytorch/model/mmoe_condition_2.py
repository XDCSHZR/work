from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.expert import MLPExpert
from layer.tower import MLPTower


class MMOE(nn.Module):
    def __init__(self, config, feature_columns):
        super(MMOE, self).__init__()
        conf = config['Model']
        self.num_experts = conf['num_experts']
        self.num_tasks = conf['num_tasks']
        self.units = conf['units']
        self.tower_hidden_units = conf['tower_hidden_units']
        self.expert_hidden_units = conf['expert_hidden_units']
        self.dropout = conf['dropout']
        self.use_bn = conf['use_bn']
        
        self.condition_1_columns, self.condition_2_columns, self.sparse_feature_columns, self.dense_feature_columns = feature_columns
        self.condition_1_feature_columns_len = len(self.condition_1_columns)
        self.condition_2_feature_columns_len = len(self.condition_2_columns)
        self.sparse_feature_columns_len = len(self.sparse_feature_columns)
        self.embedding = nn.ModuleList([nn.Embedding(feat['feat_onehot_dim'], feat['embed_dim']) for _, feat in enumerate(self.sparse_feature_columns)])
        
        self.num_feature = sum([feat['embed_dim'] for feat in self.sparse_feature_columns]) + len(self.dense_feature_columns)
        self.softmax = nn.Softmax(dim=1)

        self.expert = nn.ModuleList([MLPExpert(self.num_feature, self.units, self.expert_hidden_units, self.dropout, self.use_bn) for _ in range(self.num_experts)])
        self.gate = nn.ModuleList([nn.Linear(self.num_feature, self.num_experts) for _ in range(self.num_tasks)])
        self.gate_softmax = nn.Softmax(dim=1)
        self.towers = nn.ModuleList([MLPTower(self.units + self.condition_1_feature_columns_len + self.condition_2_feature_columns_len, 1, self.tower_hidden_units, self.dropout, self.use_bn) for _ in range(self.num_tasks)])
        # self.bn = nn.BatchNorm1d(len(self.dense_feature_columns))
        self.flatten = nn.Flatten()


    def forward(self, x):
        condition_1_inputs, condition_2_inputs, sparse_inputs, dense_inputs = x[:, :self.condition_1_feature_columns_len], x[:, self.condition_1_feature_columns_len:(self.condition_1_feature_columns_len+self.condition_2_feature_columns_len)], x[:, (self.condition_1_feature_columns_len+self.condition_2_feature_columns_len):(self.condition_1_feature_columns_len+self.condition_2_feature_columns_len+self.sparse_feature_columns_len)], x[:, (self.condition_1_feature_columns_len+self.condition_2_feature_columns_len+self.sparse_feature_columns_len):]
        sparse_inputs = sparse_inputs.to(dtype=torch.long)
        sparse_embed = torch.cat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.embedding)], dim=1)  # [batch_size, 1, embed_dim]
        sparse_embed = self.flatten(sparse_embed)
        
        # Normalization
        # dense_inputs = self.bn(dense_inputs)
        inputs = torch.cat([sparse_embed, dense_inputs], dim=1)

        inputs = inputs.to(dtype=torch.float32)
        expert = [exp(inputs) for _, exp in enumerate(self.expert)]
        expert = torch.stack(expert, dim=1)  # [batch_size, num_experts, units]
        gate = [self.gate_softmax(g(inputs)).unsqueeze(2).repeat(1,1,self.units) for _, g in enumerate(self.gate)]  # num_tasks * [batch_size, num_experts, units]

        expert = [torch.sum(g * expert, dim=1) for i, g in enumerate(gate)]  # num_tasks * [batch_size, units]
        
        tower = [t(torch.cat([expert[i], condition_1_inputs, condition_2_inputs], dim=1)) for i, t in enumerate(self.towers)]
        tower = torch.cat(tower, dim=1)  # [batch_size, num_tasks]

        out = tower
        return out
