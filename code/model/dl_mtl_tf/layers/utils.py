# -*- coding:utf-8 -*-
'''
Created on Fri Feb 11 19:00:00 2022

@author: hzr
'''

import tensorflow as tf

from tensorflow.python.keras.layers import Flatten

try:
    from tensorflow.python.ops.lookup_ops import StaticHashTable
except ImportError:
    from tensorflow.python.ops.lookup_ops import HashTable as StaticHashTable


class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None


def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)


def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


def combined_dnn_input(list_embedding_sparse, dense_value):
    sparse_dnn_input = Flatten()(concat_func(list_embedding_sparse))

    return tf.concat([tf.cast(sparse_dnn_input, tf.float32), tf.cast(dense_value,  tf.float32)], axis=-1)
