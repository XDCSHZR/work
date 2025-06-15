import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


class FocalLossV1(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, multi_class=True, reduce=True):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.multi_class = multi_class

    def forward(self, inputs, targets):
        if self.multi_class:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            if self.logits:
                BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            else:
                BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss