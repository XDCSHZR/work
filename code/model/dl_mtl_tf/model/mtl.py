# -*- coding:utf-8 -*-
'''
Created on Fri Feb 11 19:00:00 2022

@author: hzr
'''

import tensorflow as tf

from tensorflow.python.keras.models import Model

from utils.feature_column import split_features
from layers.inputs import create_embedding
from layers.core import DNN, PredictionLayer
from layers.utils import combined_dnn_input, reduce_sum


class MTL(Model):
    def __init__(self, dnn_feature_columns,
                 num_experts=3, expert_dnn_hidden_units=(256, 128, 64),
                 gate_dnn_hidden_units=(), l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                 dnn_activation='relu',
                 dnn_use_bn=False, label_types=('binary', 'binary', 'binary', 'binary'), label_names=('item_1', 'item_2', 'item_3', 'item_4')):
        super(MTL, self).__init__()
        self.orderDict_features_sparse, self.orderDict_feature_dense = split_features(dnn_feature_columns)
        self.embedding_layers = create_embedding(self.orderDict_features_sparse, l2_reg_embedding)
        self.expert_layers = [DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                                  seed=seed, name='expert_'+str(i))
                              for i in range(num_experts)]
        self.label_names = label_names
        self.gate_layers_inputs = [DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                                       seed=seed, name='gate_'+label_names[i])
                                   for i in range(len(label_names))]
        self.gate_layers_outputs = [tf.keras.layers.Dense(num_experts, use_bias=False, activation='softmax',
                                                          name='gate_softmax_'+label_names[i])
                                    for i in range(len(label_names))]
        self.logits = [tf.keras.layers.Dense(1, use_bias=False, activation=None)
                       for _ in range(len(label_names))]
        self.output_layers = [PredictionLayer(label_type, name=label_name)
                              for label_type, label_name in zip(label_types, label_names)]
        self._set_inputs(tf.TensorSpec([None, len(dnn_feature_columns)], tf.float32, name='inputs'))
    
    def call(self, inputs, training=None):
        inputs_sparse, input_dense = inputs[:, :len(self.orderDict_features_sparse)], \
                                     inputs[:, len(self.orderDict_features_sparse):]

        sparse_embedding = [emb_layer(inputs_sparse[:, i])
                            for i, emb_layer in enumerate(self.embedding_layers)]
        input_dnn = combined_dnn_input(sparse_embedding, input_dense)

        expert_outputs = [exp_layer(input_dnn) for exp_layer in self.expert_layers]
        expert_concat = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(expert_outputs)

        mmoe_outputs = []
        for i in range(len(self.label_names)):
            gate_input = self.gate_layers_inputs[i](input_dnn)
            gate_output = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))\
                (self.gate_layers_outputs[i](gate_input))
            gate_mul_expert = tf.keras.layers.Lambda(lambda x: reduce_sum(x[0]*x[1], axis=1, keep_dims=False),
                                                     name='gate_mul_expert_'+self.label_names[i])\
                ([expert_concat, gate_output])
            mmoe_outputs.append(gate_mul_expert)
            
        label_outputs = []
        for i in range(len(self.label_names)):
            logit_output = self.logits[i](mmoe_outputs[i])
            label_output = self.output_layers[i](logit_output)
            label_outputs.append(label_output)

        return label_outputs
