"""
Tower network structure.
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


# class MLPTower(nn.Module):
#     def __init__(self, input_size, output_size, hidden_units, dropout_rate=0.5, use_bn=False):
#         super(MLPTower, self).__init__()
#         if not isinstance(hidden_units, (int, list)):
#             raise ValueError("hidden_units must be a list of int or int")
#         if isinstance(hidden_units, int):
#             hidden_units = [hidden_units]
#         net = []
#         for i in range(len(hidden_units)):
#             if i == 0:
#                 net.append(("tower fc{}".format(i), nn.Linear(input_size, hidden_units[i])))
#                 if use_bn:
#                     net.append(("tower bn{}".format(i), nn.BatchNorm1d(hidden_units[i])))
#                 net.append((("tower relu{}".format(i), nn.ReLU())))
#                 net.append((("tower dropout{}".format(i), nn.Dropout(dropout_rate))))
#             else:
#                 net.append(("tower fc{}".format(i), nn.Linear(hidden_units[i - 1], hidden_units[i])))
#                 if use_bn:
#                     net.append(("tower bn{}".format(i), nn.BatchNorm1d(hidden_units[i])))
#                 net.append((("tower relu{}".format(i), nn.ReLU())))
#                 net.append((("tower dropout{}".format(i), nn.Dropout(dropout_rate))))
#         net.append(("tower fc{}".format(len(hidden_units)), nn.Linear(hidden_units[-1], output_size)))

#         self.mlptower = nn.Sequential(OrderedDict(net))


#     def forward(self, x):
#         return self.mlptower(x)

class MLPTower(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, dropout_rate=0.5, use_bn=False):
        super(MLPTower, self).__init__()
        if not isinstance(hidden_units, (int, list)):
            raise ValueError("hidden_units must be a list of int or int")
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        net = []
        if len(hidden_units) == 0:
            net.append(("tower_fc", nn.Linear(input_size, output_size)))
        else:
            for i in range(len(hidden_units)):
                if i == 0:
                    net.append(("tower_fc{}".format(i), nn.Linear(input_size, hidden_units[i])))
                    if use_bn:
                        net.append(("tower_bn{}".format(i), nn.BatchNorm1d(hidden_units[i])))
                    net.append((("tower_relu{}".format(i), nn.ReLU())))
                    net.append((("tower_dropout{}".format(i), nn.Dropout(dropout_rate))))
                else:
                    net.append(("tower_fc{}".format(i), nn.Linear(hidden_units[i - 1], hidden_units[i])))
                    if use_bn:
                        net.append(("tower_bn{}".format(i), nn.BatchNorm1d(hidden_units[i])))
                    net.append((("tower_relu{}".format(i), nn.ReLU())))
                    net.append((("tower_dropout{}".format(i), nn.Dropout(dropout_rate))))
            net.append(("tower_fc{}".format(len(hidden_units)), nn.Linear(hidden_units[-1], output_size)))

        self.mlptower = nn.Sequential(OrderedDict(net))


    def forward(self, x):
        return self.mlptower(x)