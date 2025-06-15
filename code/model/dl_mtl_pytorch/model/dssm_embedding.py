import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.dnn import DNN


class DSSM(nn.Module):
    def __init__(self, config, user_feature_columns, item_feature_columns):
        super(DSSM, self).__init__()
        
        # config
        conf = config['Model']
        
        self.user_dnn_hidden_units = conf['user_dnn_hidden_units']
        self.user_dnn_embedding = conf['user_dnn_embedding']
        
        self.item_dnn_hidden_units = conf['item_dnn_hidden_units']
        self.item_dnn_embedding = conf['item_dnn_embedding']
        
        self.dropout = conf['dropout']
        self.use_bn = conf['use_bn']
        
        self.is_eval = conf['is_eval']
        self.test_model = conf['test_model'] # 0: item_dnn_embedding, 1: user_dnn_embedding
        
        # user tower
        self.user_sparse_feature_columns, self.user_dense_feature_columns = user_feature_columns
        
        self.user_sparse_feature_columns_len = len(self.user_sparse_feature_columns)
        self.user_dense_feature_columns_len = len(self.user_dense_feature_columns)
        self.user_num_feature = sum([feat['embed_dim'] for feat in self.user_sparse_feature_columns]) + len(self.user_dense_feature_columns)
        
        self.user_embedding = nn.ModuleList([nn.Embedding(feat['feat_onehot_dim'], feat['embed_dim']) for _, feat in enumerate(self.user_sparse_feature_columns)])
        
        self.user_dnn = DNN(self.user_num_feature, self.user_dnn_embedding, self.user_dnn_hidden_units, self.dropout, self.use_bn)
        
        # item tower
        self.item_sparse_feature_columns, self.item_dense_feature_columns = item_feature_columns
        
        self.item_sparse_feature_columns_len = len(self.item_sparse_feature_columns)
        self.item_num_feature = sum([feat['embed_dim'] for feat in self.item_sparse_feature_columns]) + len(self.item_dense_feature_columns)
        
        self.item_embedding = nn.ModuleList([nn.Embedding(feat['feat_onehot_dim'], feat['embed_dim']) for _, feat in enumerate(self.item_sparse_feature_columns)])
        
        self.item_dnn = DNN(self.item_num_feature, self.item_dnn_embedding, self.item_dnn_hidden_units, self.dropout, self.use_bn)
        
        # other
        self.flatten = nn.Flatten()
        
        
    def forward(self, x):
        # input
        user_sparse_inputs, user_dense_inputs, item_sparse_inputs, item_dense_inputs = x[:, :self.user_sparse_feature_columns_len], x[:, self.user_sparse_feature_columns_len:(self.user_sparse_feature_columns_len+self.user_dense_feature_columns_len)], x[:, (self.user_sparse_feature_columns_len+self.user_dense_feature_columns_len):(self.user_sparse_feature_columns_len+self.user_dense_feature_columns_len+self.item_sparse_feature_columns_len)], x[:, (self.user_sparse_feature_columns_len+self.user_dense_feature_columns_len+self.item_sparse_feature_columns_len):]
        
        # user tower
        user_sparse_inputs = user_sparse_inputs.to(dtype=torch.long)
        
        user_sparse_embed = torch.cat([emb(user_sparse_inputs[:, i]) for i, emb in enumerate(self.user_embedding)], dim=1)
        user_sparse_embed = self.flatten(user_sparse_embed)
        
        user_inputs = torch.cat([user_sparse_embed, user_dense_inputs], dim=1)
        user_inputs = user_inputs.to(dtype=torch.float32)
        
        user_dnn_embedding = self.user_dnn(user_inputs)
        user_dnn_embedding_norm = torch.reshape(torch.norm(user_dnn_embedding, dim=-1), [-1, 1])
        user_dnn_embedding_norm_std = torch.div(user_dnn_embedding, user_dnn_embedding_norm)
        
        # item tower
        item_sparse_inputs = item_sparse_inputs.to(dtype=torch.long)
        
        item_sparse_embed = torch.cat([emb(item_sparse_inputs[:, i]) for i, emb in enumerate(self.item_embedding)], dim=1)
        item_sparse_embed = self.flatten(item_sparse_embed)
        
        item_inputs = torch.cat([item_sparse_embed, item_dense_inputs], dim=1)
        item_inputs = item_inputs.to(dtype=torch.float32)
        
        item_dnn_embedding = self.item_dnn(item_inputs)
        item_dnn_embedding_norm = torch.reshape(torch.norm(item_dnn_embedding, dim=-1), [-1, 1])
        item_dnn_embedding_norm_std = torch.div(item_dnn_embedding, item_dnn_embedding_norm)
        
        # return torch.cat([user_dnn_embedding_norm_std, item_dnn_embedding_norm_std], dim=1)
        
        if not self.training:
            if self.is_eval:
                return torch.cat([user_dnn_embedding_norm_std, item_dnn_embedding_norm_std], dim=1)
            else:
                if self.test_model == 0:
                    return item_dnn_embedding_norm_std
                else:
                    return user_dnn_embedding_norm_std
        else:
            return torch.cat([user_dnn_embedding_norm_std, item_dnn_embedding_norm_std], dim=1)
