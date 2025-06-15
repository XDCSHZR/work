from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.expert import MLPExpert
from layer.tower import MLPTower
from layer.shared_bottom_unit import SharedBottomUnit
from layer.mmoe_unit import MMOEUnit
from layer.ple_unit import PLEUnit
from layer.attention import Attention, Info


class MFH(nn.Module):
    def __init__(self, config, feature_columns):
        super(MFH, self).__init__()
        # params
        # self.num_feature = config.num_feature
        conf = config.Model
        self.conf = conf
        self.conf_level0 = conf['level0']
        self.conf_level1 = conf['level1']
        self.conf_level2 = conf['level2']
        self.conf_level0_num_tasks = self.conf_level0['num_tasks']

        self.dense_feature_columns, self.sparse_feature_columns, self.condition_columns = feature_columns
        self.sparse_feature_columns_len = len(self.sparse_feature_columns)
        self.condition_feature_columns_len = len(self.condition_columns)
        self.embedding = nn.ModuleList([nn.Embedding(feat['feat_onehot_dim'], feat['embed_dim']) for _, feat in
                                        enumerate(self.sparse_feature_columns)])

        self.num_feature = sum([feat['embed_dim'] for feat in self.sparse_feature_columns]) + len(
            self.dense_feature_columns)

        self.level0 = SharedBottomUnit(self.num_feature, self.conf_level0['output_units'], self.conf_level0['hidden_units'])

        self.level1 = nn.ModuleList([MMOEUnit(v) for _, v in self.conf_level1.items()])

        self.level2 = nn.ModuleList([MMOEUnit(v) for _, v in self.conf_level2.items()])

        self.towers_head = nn.ModuleList(
            [MLPTower(conf.tower['input_units'], 1, conf.tower['hidden_units'], conf.tower['dropout'],
                      conf.tower['use_bn']) for _ in
             range(sum([i['num_tasks'] for i in self.conf_level2.values()])//3)])

        # print(len(self.dense_feature_columns))
        self.bn = nn.BatchNorm1d(len(self.dense_feature_columns))
        self.flatten = nn.Flatten()

    def forward(self, x):
        condition_inputs, sparse_inputs, dense_inputs = x[:, :self.condition_feature_columns_len], x[:, self.condition_feature_columns_len:(self.condition_feature_columns_len + self.sparse_feature_columns_len)], x[:,(self.condition_feature_columns_len + self.sparse_feature_columns_len):]
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
        inputs = inputs.to(dtype=torch.float32)
        level0_outputs = self.level0(inputs)
        level0_outputs = [level0_outputs for i in range(self.conf_level0_num_tasks)]

        level1_outputs = [l1(level0_outputs[i]) for i, l1 in enumerate(self.level1)]  # (2, 3, 4)
        # level1_outputs = [item for sublist in level1_outputs for item in sublist]  # flatten
        # flatten
        _level1_outputs = []
        for i in level1_outputs:
            _level1_outputs += i
        
        level2_outputs = [l2(_level1_outputs[i]) for i, l2 in enumerate(self.level2)]  # (12, 12, 8, 8, 8, 6, 6, 6, 6)
        towers_head_inputs_0 = level2_outputs[0] + level2_outputs[1]
        towers_head_inputs_1 = level2_outputs[2][:4] + level2_outputs[3][:4] + level2_outputs[4][:4] + \
                               level2_outputs[2][4:] + level2_outputs[3][4:] + level2_outputs[4][4:]
        towers_head_inputs_2 = [[level2_outputs[5:][j][i] for j in range(len(level2_outputs[5:]))] for i in
                                range(len(level2_outputs[5:][0]))]
        # towers_head_inputs_2 = [item for sublist in towers_head_inputs_2 for item in sublist]  # flatten
        # flatten
        _towers_head_inputs_2 = []
        for i in towers_head_inputs_2:
            _towers_head_inputs_2 += i
        
        # print(len(towers_head_inputs_0))
        # print(len(towers_head_inputs_1))
        # print(len(towers_head_inputs_2))
        # print(len(self.towers_head))
        
        towers_out = [th(torch.cat([towers_head_inputs_0[i] + towers_head_inputs_1[i] + _towers_head_inputs_2[i], condition_inputs], dim=1)) for i, th in enumerate(self.towers_head)]

        out = torch.cat(towers_out, dim=1)  # [batch_size, num_tasks]
        # print(out.shape)

        return out

    def loss(self, out, label, weight=[1.1, 1.1, 1.1, 1.2, 1, 1, 1, 1, 0.9, 0.9, 0.9, 0.9, 1.2, 1.2, 1.2, 1.3, 1.1, 1.1, 1.1, 1.2, 0.9, 0.9, 0.9, 0.9], mask=-100): # mask
        y_pred = torch.sigmoid(out)  # (batch_size, 24)
        y_true = label    # (batch_size, 24)
        _mask = y_true.ne(mask)

        # loss = F.binary_cross_entropy(y_pred, y_true, reduction='none') * _mask
        # print(y_pred.shape, y_true.shape)
        loss_list = [torch.mean(F.binary_cross_entropy(y_pred[:, i], y_true[:, i], reduction='none') * _mask[:, i]) for i in range(y_true.shape[-1])]
        loss_list = [loss_list[i] * weight[i] for i in range(len(loss_list))]

        return loss_list


