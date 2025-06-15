import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWL_weighted_multi_task(nn.Module):
    def __init__(self):
        super(BCEWL_weighted_multi_task, self).__init__()
        
    def forward(self, inputs, targets, weights):
        loss_BCEWL = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 过滤batch中权重为0的样本
        # weights_nozero_index = torch.nonzero(weights)
        # loss_BCEWL_weighted = loss_BCEWL[weights_nozero_index].reshape(-1) * weights[weights_nozero_index].reshape(-1)
        
        loss_BCEWL_weighted = loss_BCEWL * weights
        
        loss = torch.mean(loss_BCEWL_weighted)
        
        return loss
