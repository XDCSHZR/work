"""
Imputation network structure.
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPImputation(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, dropout_rate=0.5, use_bn=False):
        super(MLPImputation, self).__init__()
        if not isinstance(hidden_units, (int, list)):
            raise ValueError("hidden_units must be a list of int or int")
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        net = []
        # net.append(("imputation bn{}".format(0), nn.BatchNorm1d(input_size)))
        for i in range(len(hidden_units)):
            if i == 0:
                net.append(("imputation_fc{}".format(i), nn.Linear(input_size, hidden_units[i])))
                if use_bn:
                    net.append(("imputation_bn{}".format(i), nn.BatchNorm1d(hidden_units[i])))
                net.append((("imputation_relu{}".format(i), nn.ReLU())))
                net.append((("imputation_dropout{}".format(i), nn.Dropout(dropout_rate))))
            else:
                net.append(("imputation_fc{}".format(i), nn.Linear(hidden_units[i - 1], hidden_units[i])))
                if use_bn:
                    net.append(("imputation_bn{}".format(i), nn.BatchNorm1d(hidden_units[i])))
                net.append((("imputation_relu{}".format(i), nn.ReLU())))
                net.append((("imputation_dropout{}".format(i), nn.Dropout(dropout_rate))))
        net.append(("imputation_fc{}".format(len(hidden_units)), nn.Linear(hidden_units[-1], output_size)))

        self.mlpimputation = nn.Sequential(OrderedDict(net))


    def forward(self, x):
        return self.mlpimputation(x)



