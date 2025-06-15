import os
from re import L
import numpy as np
import utils.utils as util
import torch
from utils.dataset import *
from model.adatt import *
from layer.attention import Attention, Info
from loss.focalloss import FocalLossV1 as FocalLoss
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = util.get_logger('')
best_loss = np.inf
best_auc = 0.6


def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}


def denseFeature(feat):
    return {'feat': feat}


def train(train_loader, test_loader, model, optimizer, scheduler, criterion, epochs, mode, stage):
    # stage = [5, 10, 15]
    assert isinstance(stage, list) == 1
    stage = [0] + stage
    # print(stage)
    for epoch in range(epochs):
        model.train()
        if epoch < stage[-1]:
            for s in range(len(stage) - 1):
                if epoch < stage[s + 1] and epoch >= stage[s]:
                    for i, (x, y) in enumerate(train_loader):
                        x, y = x.to(device).to(torch.float32), y.to(device).to(torch.float32)
                        x, y = x.reshape((-1, x.shape[-1])), y.reshape((-1, y.shape[-1]))
                        optimizer.zero_grad()
                        output = model(x)
                        loss_list = model.loss(output, y, x)[0]
                        # print('loss0', [ddd.item() for ddd in model.loss(output, y)[1]])
                        # print('loss1',[ddd.item() for ddd in model.loss(output, y)[2]])
                        loss = sum(loss_list[:s+1])
                        # loss = loss_list[s]
                        loss.backward()
                        optimizer.step()
                        if i % 100 == 0:
                            str_loss = ' '.join(['Loss{}: {:.4f}'.format(i, loss_list[i]) for i in range(len(loss_list))])
                            logger.info('Epoch: [{}/{}], Step: [{}/{}], Lr: {:.6f}, {}, Loss: {:.6f}'.format(
                            epoch + 1, epochs, i + 1, len(train_loader), optimizer.param_groups[0]['lr'], str_loss, loss.item()))
                        # if i % 5000 == 0:
                        #     val(test_loader, model, criterion, mode, output.shape[-1])

        else:
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device).to(torch.float32), y.to(device).to(torch.float32)
                x, y = x.reshape((-1, x.shape[-1])), y.reshape((-1, y.shape[-1]))
                optimizer.zero_grad()
                output = model(x)
                # loss_list = [criterion[i](output[:, i], y[:, i]) for i in range(output.shape[-1])]
                loss_list = model.loss(output, y, x)[0]
                loss = sum(loss_list)
                # loss = sum(loss_list) + 10 * loss_list[-1]
                # loss = loss_list[-1]
                
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    str_loss = ' '.join(['Loss{}: {:.4f}'.format(i, loss_list[i]) for i in range(len(loss_list))])
                    logger.info('Epoch: [{}/{}], Step: [{}/{}], Lr: {:.6f}, {}, Loss: {:.6f}'.format(
                    epoch + 1, epochs, i + 1, len(train_loader), optimizer.param_groups[0]['lr'], str_loss, loss.item()))
                # if i % 5000 == 0:
                #     val(test_loader, model, criterion, mode, output.shape[-1])
                    
        logger.info('Epoch: [{}/{}] Test:'.format(epoch+1, epochs))
        val(test_loader, model, criterion, mode, output.shape[-1])
        print('loss0', [ddd.item() for ddd in model.loss(output, y, x)[1]])
        print('loss1',[ddd.item() for ddd in model.loss(output, y, x)[2]])

        scheduler.step()



def val(test_loader, model, criterion, mode, task_num):
    model.eval()
    global best_loss
    global best_auc
    test_loss = 0
    test_loss0 = 0
    test_loss1 = 0
    y_pred = []
    test_loss_list = np.array([0.0 for _ in range(task_num)])
    
    # multi-label
    if mode == 'multi-label':
        # y_true = data_test[1]
        y_true = []
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device).to(torch.float32), y.to(device).to(torch.float32)
                x, y = x.reshape((-1, x.shape[-1])), y.reshape((-1, y.shape[-1]))
                output = model(x)
                output1 = torch.sigmoid(output)
                # test_loss += criterion(output, y).item()
                # test_loss0 = test_loss0 + criterion[0](output[:,0], y[:,0]).item()
                # test_loss1 = test_loss1 + criterion[1](output[:,1], y[:,1]).item()
                # test_loss = test_loss + criterion[0](output[:,0], y[:,0]).item() + criterion[1](output[:,1], y[:,1]).item()
                
                # test_loss_list += np.array([criterion[i](output[:,i], y[:,i]).item() for i in range(output.shape[-1])])
                test_loss_list += np.array([i.item() for i in model.loss(output, y, x)[0]])
                test_loss = sum(test_loss_list)
                
                if i == 0:
                    y_pred = output1.cpu().numpy()
                    y_true = y.cpu().numpy()
                else:
                    y_pred = np.concatenate((y_pred, output1.cpu().numpy()), axis=0)
                    y_true = np.concatenate((y_true, y.cpu().numpy()), axis=0)
    # multi-class
    else:
        y_true = OneHotEncoder().fit_transform(data_test[1].reshape(-1, 1)).toarray()
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device).to(torch.float32), y.to(device).to(torch.long)
                output = model(x)
                # output = F.softmax(output, dim=1)
                test_loss += criterion(output, y, x).item()
                if i == 0:
                    y_pred = output.cpu().numpy()
                else:
                    y_pred = np.concatenate((y_pred, output.cpu().numpy()), axis=0)
    
    
    # test_loss_list = list(test_loss_list/(len(test_loader)*256))
    # test_loss /= len(test_loader)*256
    
    test_loss_list = list(test_loss_list/len(test_loader))
    test_loss /= len(test_loader)
    
    
    num_class = y_true.shape[1]
    test_auc = [util.auc(y_true[:, i], y_pred[:, i]) for i in range(num_class)]
    str_auc = ' '.join(['AUC{}: {:.4f}'.format(i, test_auc[i]) for i in range(num_class)])
    str_loss = ' '.join(['Average loss{}: {:.4f}'.format(i, test_loss_list[i]) for i in range(num_class)])

    logger.info('Test set: Average loss: {:.4f}, {}, {}'.format(test_loss, str_loss, str_auc))
    # logger.info('Test set: Average loss: {:.4f}, AUC: {:.4f}\n'.format(test_loss, test_auc))
    # if test_loss < best_loss:
    #     best_loss = test_loss
    #     torch.save(model.state_dict(), './modelSaved/model.pth')
    #     model1 = torch.jit.script(model)
    #     torch.jit.save(model1, './modelSaved/mmoe.pt')
    #     logger.info('Save model with loss: {:.4f}, {}'.format(test_loss, str_auc))
    
    if test_auc[-1] > best_auc: # and test_auc[0] > 0.81 and test_auc[1] > 0.83:
        best_auc = test_auc[-1]
        torch.save(model.state_dict(), './modelSaved/adatt_sp_v1_0.pth')
        model1 = torch.jit.script(model)
        torch.jit.save(model1, './modelSaved/adatt_sp_v1_0.pt')
        logger.info('Save model with loss: {:.4f}, {}, {}'.format(test_loss, str_loss, str_auc))


if __name__ == '__main__':
    # get configuration
    __CONF_PATH = './config/adatt_sp.yaml'
    config = util.get_config(__CONF_PATH)
    util.seed_everything(config.Seed)

    logger.info(config.Model)
    logger.info(config.Train)

    # get data
    # x, y, feature_columns = data_preprocess(config)
    # X_train, X_test, y_train, y_test = util.split_data(x, y, ratio=0.2, shuffle=True)
    train_loader = DataLoader(NpDatasetLoader(config.Data.filePath, 'train', [0,1,2,3]), config.Train.batchSize, shuffle=True, num_workers=config.Train.numWorkers)
    eval_loader = DataLoader(NpDatasetLoader(config.Data.filePath, 'val', [0,1,2,3]), config.Train.batchSize, shuffle=False, num_workers=config.Train.numWorkers)
    
    # feature_columns = get_columns(config.Data.trainFilePath)
    # train_loader = DataLoader(PandasDatasetIterator(config.Data.trainFilePath, process), config.Train.batchSize, num_workers=config.Train.numWorkers)
    # eval_loader = DataLoader(PandasDatasetIterator(config.Data.testFilePath, process), config.Train.batchSize, num_workers=config.Train.numWorkers)
    

    feature_columns_np = list(np.load(config.Data.featureColumns, allow_pickle=True))
    feature_columns = []
    # replace embed dim
    for i in range(len(feature_columns_np)):
        if i == 1:
            tmp = []
            for j in feature_columns_np[i]:
                j['embed_dim'] = config.Model['embed_dim']
                tmp.append(j)
            feature_columns.append(tmp)
        else:
            feature_columns.append(feature_columns_np[i])
    
    # print(feature_columns)
    
    model = AdaTTSp(config, feature_columns).to(device)
    # model = AdaTTWSharedExps(config, feature_columns).to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):  #(nn.Conv2d, nn.Linear)
            nn.init.xavier_uniform_(m.weight)
            # nn.init.kaiming_uniform_(m.weight)
        
        elif isinstance(m, nn.Conv2d):  #(nn.Conv2d, nn.Linear)
            nn.init.kaiming_uniform_(m.weight)   
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, Attention):
            nn.init.xavier_uniform_(m.q_layer.weight)
            nn.init.constant_(m.q_layer.bias, 0)
            nn.init.xavier_uniform_(m.k_layer.weight)
            nn.init.constant_(m.k_layer.bias, 0)
            nn.init.xavier_uniform_(m.v_layer.weight)
            nn.init.constant_(m.v_layer.bias, 0)
        
        elif isinstance(m, Info):
            # print(m.info.info_fc)
            nn.init.constant_(m.info.info_fc.weight, 0)
            nn.init.constant_(m.info.info_fc.bias, 0)
        
                
    # multi-class
    if config.Model['mode'] == 'multi-class':
        if config.Model['loss'] == 'focalloss':
            criterion = FocalLoss(multi_class=True).to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)   # softmax-log-NLLLoss
        
    
    # multi-label
    if config.Model['mode'] == 'multi-label':
        if config.Model['loss'] == 'focalloss':
            criterion = [nn.BCEWithLogitsLoss().to(device) for _ in range(config.Model['num_tasks'] - 1)] + [FocalLoss(multi_class=False).to(device)]
            # criterion = (nn.BCEWithLogitsLoss().to(device), FocalLoss(multi_class=False).to(device))
        else:
            criterion = [nn.BCEWithLogitsLoss().to(device) for _ in range(config.Model['num_tasks'])]
            # criterion = (nn.BCEWithLogitsLoss().to(device), nn.BCEWithLogitsLoss().to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Train.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.Train.lr, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.Train.stepSize, gamma=config.Train.gamma)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.Train.milestones, gamma=config.Train.gamma)
    train(train_loader, eval_loader, model, optimizer, scheduler, criterion, config.Train.epochs, config.Model['mode'], config.Train['stage'])
    # train(train_loader, eval_loader, model, optimizer, scheduler, criterion, config.Train.epochs, config.Model['mode'])



