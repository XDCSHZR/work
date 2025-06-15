# -*- coding:utf-8 -*-
'''
Created on Fri Feb 11 19:00:00 2022

@author: hzr
'''

from collections import namedtuple, OrderedDict
from tensorflow.python.keras.initializers import RandomNormal, Zeros

DEFAULT_GROUP_NAME = 'default_group'


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'vocabulary_path', 'dtype',
                             'embeddings_initializer', 'embedding_name', 'group_name', 'trainable'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, vocabulary_path=None, dtype='int32',
                embeddings_initializer=None, embedding_name=None, group_name=DEFAULT_GROUP_NAME, trainable=True):
        if embedding_dim == 'auto':
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2020)
        if embedding_name is None:
            embedding_name = name

        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, vocabulary_path,
                                              dtype,
                                              embeddings_initializer,
                                              embedding_name, group_name, trainable)

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat',
                           ['name', 'dimension', 'dtype', 'transform_fn'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype='float32', transform_fn=None):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype, transform_fn)

    def __hash__(self):
        return self.name.__hash__()


def split_features(feature_columns):
    orderDict_sparse_features = OrderedDict()
    orderDict_dense_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            orderDict_sparse_features[fc.name] = fc
        elif isinstance(fc, DenseFeat):
            orderDict_dense_features[fc.name] = fc
        else:
            raise TypeError('Invalid feature column type,got', type(fc))

    return orderDict_sparse_features, orderDict_dense_features


class DenseFeat_Condition(namedtuple('DenseFeat_Condition', 
                                     ['name', 'dimension', 'dtype', 'transform_fn'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype='float32', transform_fn=None):
        return super(DenseFeat_Condition, cls).__new__(cls, name, dimension, dtype, transform_fn)

    def __hash__(self):
        return self.name.__hash__()


def split_features_condition(feature_columns):
    orderDict_dense_features_condition = OrderedDict()
    orderDict_sparse_features = OrderedDict()
    orderDict_dense_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, DenseFeat_Condition):
            orderDict_dense_features_condition[fc.name] = fc
        elif isinstance(fc, SparseFeat):
            orderDict_sparse_features[fc.name] = fc
        elif isinstance(fc, DenseFeat):
            orderDict_dense_features[fc.name] = fc
        else:
            raise TypeError('Invalid feature column type,got', type(fc))

    return orderDict_dense_features_condition, orderDict_sparse_features, orderDict_dense_features
