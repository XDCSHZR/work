import torch
import torch.nn as nn
import torch.nn.functional as F


class IPSLoss(nn.Module):
    def __init__(self, ips_weights=0.1, task_weights=1.0, activation=True):
        super(IPSLoss, self).__init__()
        assert isinstance(ips_weights, (list, int, float))
        assert isinstance(task_weights, (list, int, float))
        self.ips_weights = ips_weights
        self.task_weights = task_weights
        self.activation = activation

    def forward(self, inputs, targets):
        if self.activation:
            inputs = torch.sigmoid(inputs)
        num_tasks = targets.shape[-1]
        assert num_tasks >= 2
        if not isinstance(self.ips_weights, list):
            ips_weight = [self.ips_weights for i in range(num_tasks - 1)]
        else:
            assert len(self.ips_weights) + 1 == num_tasks
            ips_weight = self.ips_weights

        if not isinstance(self.task_weights, list):
            task_weight = [self.task_weights for i in range(num_tasks)]
        else:
            assert len(self.task_weights) == num_tasks
            task_weight = self.task_weights

        _ptask = []
        _label = []
        _out = []
        p_tmp = torch.ones_like(targets[:, 0])
        min_tensor = 0.0000001 * torch.ones_like(targets[:, 0])
        loss_ips = []
        loss_task = []
        for i in range(num_tasks):
            p_tmp = torch.maximum(torch.mul(p_tmp, inputs[:, i]), min_tensor)
            _ptask.append(p_tmp)
            _label.append(targets[:, i])
            _out.append(inputs[:, i])
            loss_task.append(F.binary_cross_entropy(p_tmp, targets[:, i]))

        for i in range(1, num_tasks):
            ips = torch.clamp(torch.reciprocal(_out[i - 1]), min=-1e3, max=1e3)
            _loss = F.binary_cross_entropy(inputs[:, i], targets[:, i], reduction='none')
            loss_ips.append(torch.mean(torch.mul(torch.mul(_loss, _label[i - 1]), ips)))

        loss_ips = [loss_ips[i] * ips_weight[i] for i in range(len(loss_ips))]
        loss_task = [loss_task[i] * task_weight[i] for i in range(len(loss_task))]
        return sum(loss_ips) + sum(loss_task), loss_ips, loss_task


class DRLoss(nn.Module):
    def __init__(self, ips_weights=0.1, task_weights=1.0, activation=True):
        super(DRLoss, self).__init__()
        assert isinstance(ips_weights, (list, int, float))
        assert isinstance(task_weights, (list, int, float))
        self.ips_weights = ips_weights
        self.task_weights = task_weights
        self.activation = activation

    def forward(self, inputs, targets):
        num_tasks = targets.shape[-1]
        assert num_tasks >= 2
        if self.activation:
            inputs = torch.cat([torch.sigmoid(inputs[:, :num_tasks]), inputs[:, num_tasks:]], dim=1)

        if not isinstance(self.ips_weights, list):
            ips_weight = [self.ips_weights for i in range(num_tasks - 1)]
        else:
            assert len(self.ips_weights) + 1 == num_tasks
            ips_weight = self.ips_weights

        if not isinstance(self.task_weights, list):
            task_weight = [self.task_weights for i in range(num_tasks)]
        else:
            assert len(self.task_weights) == num_tasks
            task_weight = self.task_weights

        _ptask = []
        _label = []
        _out = []
        p_tmp = torch.ones_like(targets[:, 0])
        min_tensor = 0.0000001 * torch.ones_like(targets[:, 0])
        loss_dr = []
        loss_task = []
        for i in range(num_tasks):
            p_tmp = torch.maximum(torch.mul(p_tmp, inputs[:, i]), min_tensor)
            _ptask.append(p_tmp)
            _label.append(targets[:, i])
            _out.append(inputs[:, i])
            loss_task.append(F.binary_cross_entropy(p_tmp, targets[:, i]))

        for i in range(1, num_tasks):
            imp_out = inputs[:, i + num_tasks - 1]
            ips = torch.clamp(torch.reciprocal(_out[i - 1]), min=-1e3, max=1e3)
            err = torch.subtract(F.binary_cross_entropy(inputs[:, i], targets[:, i], reduction='none'), imp_out)
            loss_dr_err = imp_out + torch.mul(torch.mul(err, _label[i - 1]), ips)
            loss_dr_imp = torch.mul(torch.mul(torch.square(err), _label[i - 1]), ips)
            loss_dr.append(torch.mean(loss_dr_err + loss_dr_imp))

        loss_dr = [loss_dr[i] * ips_weight[i] for i in range(len(loss_dr))]
        loss_task = [loss_task[i] * task_weight[i] for i in range(len(loss_task))]
        return sum(loss_dr) + sum(loss_task), loss_dr, loss_task

class MIPSLoss(nn.Module):
    """
    IPS loss for mmoe.
    """
    def __init__(self, ips_weights=0.1, task_weights=1.0, activation=True):
        super(MIPSLoss, self).__init__()
        assert isinstance(ips_weights, (list, int, float))
        assert isinstance(task_weights, (list, int, float))
        self.ips_weights = ips_weights
        self.task_weights = task_weights
        self.activation = activation

    def forward(self, inputs, targets):
        if self.activation:
            inputs = torch.sigmoid(inputs)
        num_tasks = targets.shape[-1]
        assert num_tasks >= 2
        if not isinstance(self.ips_weights, list):
            ips_weight = [self.ips_weights for i in range(num_tasks - 1)]
        else:
            assert len(self.ips_weights) + 1 == num_tasks
            ips_weight = self.ips_weights

        if not isinstance(self.task_weights, list):
            task_weight = [self.task_weights for i in range(num_tasks)]
        else:
            assert len(self.task_weights) == num_tasks
            task_weight = self.task_weights

        _ptask = []
        _label = []
        _out = []
        min_tensor = 0.000001 * torch.ones_like(targets[:, 0])
        max_tensor = torch.ones_like(targets[:, 0])
        loss_ips = []
        loss_task = []
        for i in range(num_tasks):
            if i > 0:
                p_tmp = torch.minimum(torch.div(inputs[:, i], inputs[:, i-1]), max_tensor)
                p_tmp = torch.maximum(p_tmp, min_tensor)
                _ptask.append(p_tmp)
            _label.append(targets[:, i])
            _out.append(inputs[:, i])
            loss_task.append(F.binary_cross_entropy(inputs[:, i], targets[:, i]))

        for i in range(1, num_tasks):
            ips = torch.clamp(torch.reciprocal(_out[i - 1]), min=-1e5, max=1e5)
            _loss = F.binary_cross_entropy(_ptask[i - 1], _label[i], reduction='none')
            loss_ips.append(torch.mean(torch.mul(torch.mul(_loss, _label[i - 1]), ips)))

        loss_ips = [loss_ips[i] * ips_weight[i] for i in range(len(loss_ips))]
        loss_task = [loss_task[i] * task_weight[i] for i in range(len(loss_task))]
        return sum(loss_ips) + sum(loss_task), loss_ips, loss_task



def top3_recall(df):
    score_cols = ['product0_score', 'product1_score', 'product2_score', 'product3_score', 'product4_score', 'product5_score']
    product_cols = ['t1.product0', 't1.product1', 't1.product2', 't1.product3', 't1.product4', 't1.product5']
    label = ['y']
    df['top3'] = df.apply(lambda row: sorted(dict(row).items(), key=lambda x: x[1])[-1], axis=1)
