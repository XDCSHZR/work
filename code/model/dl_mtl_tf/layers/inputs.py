# -*- coding:utf-8 -*-
'''
Created on Fri Feb 11 19:00:00 2022

@author: hzr
'''

from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2


def create_embedding(sparse_feature_columns, l2_reg, prefix='sparse_'):
    list_embedding_sparse = []
    for k, v in sparse_feature_columns.items():
        emb = Embedding(v.vocabulary_size, v.embedding_dim,
                        embeddings_initializer=v.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix+'_emb_'+v.embedding_name)
        emb.trainable = v.trainable
        list_embedding_sparse.append(emb)

    return list_embedding_sparse
