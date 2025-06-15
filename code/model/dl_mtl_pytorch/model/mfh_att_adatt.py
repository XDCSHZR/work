from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.expert import MLPExpert, MLP
from layer.tower import MLPTower
from layer.shared_bottom_unit import SharedBottomUnit
from layer.mmoe_unit import MMOEUnit
from layer.ple_unit import PLEUnit
from layer.adatt_unit import AdaTTSpUnit
from layer.attention import Attention, Info


class MFHATTAdaTT(nn.Module):
    def __init__(self, config, feature_columns):
        super(MFHATTAdaTT, self).__init__()
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

        # self.level2 = nn.ModuleList([MMOEUnit(v) for _, v in self.conf_level2.items()])
        self.level2 = nn.ModuleList([AdaTTSpUnit(v) for _, v in self.conf_level2.items()])

        self.towers = nn.ModuleList([MLPTower(conf.tower['input_units'], conf.tower['output_units'], conf.tower['hidden_units'], conf.tower['dropout'], conf.tower['use_bn']) for _ in range(sum([i['num_tasks'] for i in self.conf_level2.values()])//3)])
        
        self.infos = nn.ModuleList([Info(conf.tower['output_units'], conf.info['dropout'], conf.info['use_bn']) for _ in range(sum([i['num_tasks'] for i in self.conf_level2.values()])//3 - 6)])
        
        self.aits = nn.ModuleList([Attention(conf.tower['output_units'], bias=True) for _ in range(sum([i['num_tasks'] for i in self.conf_level2.values()])//3 - 6)])
        
        self.towers_head = nn.ModuleList([MLPTower(conf.head['input_units'], 1, conf.head['hidden_units'], conf.head['dropout'], conf.head['use_bn']) for _ in range(sum([i['num_tasks'] for i in self.conf_level2.values()])//3)])

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
        
        towers_out = [th(torch.cat([towers_head_inputs_0[i] + towers_head_inputs_1[i] + _towers_head_inputs_2[i], condition_inputs], dim=1)) for i, th in enumerate(self.towers)]
        
        towers_out = [towers_out[4*i:4*(i+1)] for i in range(len(towers_out)//4)]
        info = [f(towers_out[i//3][i%3]) for i, f in enumerate(self.infos)]
        
        # print(towers_out[0][0].shape, info[0].shape)
        # print(towers_out[0][0].unsqueeze(1).shape, info[0].unsqueeze(1).shape)
        
        ait = [a(torch.cat([info[i].unsqueeze(1), towers_out[i//3][i%3+1].unsqueeze(1)], dim=1)) for i, a in enumerate(self.aits)]
        
        heads_out = [h(towers_out[i//4][i%4]) if i%4 == 0 else h(ait[(i//4)*3+(i%4-1)]) for i, h in enumerate(self.towers_head)]        

        out = torch.cat(heads_out, dim=1)  # [batch_size, num_tasks]
        # print(out.shape)

        return out
    
    def loss(self, out, label, mask=-100, weight=0.6, loss_weight=[2, 2, 2, 2, 3, 3, 3, 3, 0.5, 0.5, 0.5, 0.5, 5, 5, 5, 5, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]):        
        y_pred = torch.sigmoid(out)  # (batch_size, 24)
        y_true = label    # (batch_size, 24)
        _mask = y_true.ne(mask)

        # loss = F.binary_cross_entropy(y_pred, y_true, reduction='none') * _mask
        # print(y_pred.shape, y_true.shape)
        loss_list = [torch.mean(F.binary_cross_entropy(y_pred[:, i], y_true[:, i], reduction='none') * _mask[:, i]) for i in range(y_true.shape[-1])]
        
        _y_pred = y_pred * _mask

        constraint_loss = []
        for i in range(y_true.shape[-1]//4):
            for j in range(1, 4):
                label_constraint = torch.maximum(_y_pred[:, 4*i+j] - _y_pred[:, 4*i+j-1], torch.zeros_like(_y_pred[:, 4*i+j]))
                constraint_loss.append(torch.sum(label_constraint)/label_constraint.shape[0])
        
        # loss = sum(loss_list) + weight * sum(constraint_loss)
        loss = [loss_list[i] if i%4 == 0 else loss_list[i] + weight * constraint_loss[(i//4)*3+(i%4-1)] for i in range(len(loss_list))]
        loss = [loss[i] * loss_weight[i] for i in range(len(loss))]
        
        return (loss, loss_list, constraint_loss)


