"""
DNN structure.
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, dropout_rate=0.5, use_bn=False):
        super(DNN, self).__init__()
        if not isinstance(hidden_units, (int, list)):
            raise ValueError("hidden_units must be a list of int or int")
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        net = []
        # net.append(("dnn bn{}".format(0), nn.BatchNorm1d(input_size)))
        for i in range(len(hidden_units)):
            if i == 0:
                net.append(("dnn fc{}".format(i), nn.Linear(input_size, hidden_units[i])))
                if use_bn:
                    net.append(("dnn bn{}".format(i), nn.BatchNorm1d(hidden_units[i])))
                net.append((("dnn relu{}".format(i), nn.ReLU())))
                net.append((("dnn dropout{}".format(i), nn.Dropout(dropout_rate))))
            else:
                net.append(("dnn fc{}".format(i), nn.Linear(hidden_units[i - 1], hidden_units[i])))
                if use_bn:
                    net.append(("dnn bn{}".format(i), nn.BatchNorm1d(hidden_units[i])))
                net.append((("dnn relu{}".format(i), nn.ReLU())))
                net.append((("dnn dropout{}".format(i), nn.Dropout(dropout_rate))))
        net.append(("dnn fc{}".format(len(hidden_units)), nn.Linear(hidden_units[-1], output_size)))

        self.dnn = nn.Sequential(OrderedDict(net))


    def forward(self, x):
        return self.dnn(x)



