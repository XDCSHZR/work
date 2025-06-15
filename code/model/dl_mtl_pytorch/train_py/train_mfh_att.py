import os
from re import L
import numpy as np
import utils.utils as util
import torch
from utils.dataset import *
from model.mfh_att import *
from layer.attention import Attention, Info
from loss.focalloss import FocalLossV1 as FocalLoss
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = util.get_logger('')
best_loss = np.inf
best_auc = 0.5

def train(train_loader, test_loader, model, optimizer, scheduler, epochs, mode, stage):
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
                        loss_list = model.loss(output, y)[0]
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
                            
        else:
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device).to(torch.float32), y.to(device).to(torch.float32)
                x, y = x.reshape((-1, x.shape[-1])), y.reshape((-1, y.shape[-1]))
                optimizer.zero_grad()
                output = model(x)
                loss_list, constrait_loss = model.loss(output, y)[0], model.loss(output, y)[2]
                loss = sum(loss_list)
                # loss = sum(loss_list) + 10 * loss_list[-1]
                # loss = loss_list[-1]
                
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    str_loss = ' '.join(['Loss{}: {:.4f}'.format(i+1, loss_list[i]) for i in range(len(loss_list))])
                    logger.info('Epoch: [{}/{}], Step: [{}/{}], Lr: {:.6f}, {}, Loss: {:.6f}'.format(
                    epoch + 1, epochs, i + 1, len(train_loader), optimizer.param_groups[0]['lr'], str_loss, loss.item()))
                    
                    print('loss ait',[ddd.item() for ddd in constrait_loss])
                    
                    # model1 = torch.jit.script(model)
                    # torch.jit.save(model1, './modelSaved/mfh_v1_0.pt')
                    # print('model saved')


        logger.info('Epoch: [{}/{}] Test:'.format(epoch+1, epochs))
        val(test_loader, model, mode, output.shape[-1], epoch)
        print('loss ait',[ddd.item() for ddd in constrait_loss])
        
        scheduler.step()



def val(test_loader, model, mode, task_num, epoch):
    model.eval()
    global best_loss
    global best_auc
    test_loss = 0
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

                test_loss_list += np.array([i.item() for i in model.loss(output, y)[0]])
                test_loss = sum(test_loss_list)
                
                if i == 0:
                    y_pred = output1.cpu().numpy()
                    y_true = y.cpu().numpy()
                else:
                    y_pred = np.concatenate((y_pred, output1.cpu().numpy()), axis=0)
                    y_true = np.concatenate((y_true, y.cpu().numpy()), axis=0)
    
    
    test_loss_list = list(test_loss_list/len(test_loader))
    test_loss /= len(test_loader)
    
    
    num_class = y_true.shape[1]
    test_auc = [util.auc(y_true[:, i][y_true[:, i] != -100], y_pred[:, i][y_true[:, i] != -100]) for i in range(num_class)]
    name_list = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff']
    str_auc = 'Epoch ' + str(epoch+1) + '\n'
    str_loss = 'Epoch ' + str(epoch+1) + '\n'
    for i in range(num_class // 4):
        tmp_auc = test_auc[i*4:(i+1)*4]
        tmp_auc_str = tmp_loss_str = name_list[i] + ': '
        tmp_auc_str += ' '.join(['AUC{}: {:.4f}'.format(i+1, tmp_auc[i]) for i in range(4)])
        str_auc = str_auc + tmp_auc_str + '\n'
        
        tmp_loss = test_loss_list[i*4:(i+1)*4]
        tmp_loss_str += ' '.join(['Average loss{}: {:.4f}'.format(i+1, tmp_loss[i]) for i in range(4)])
        str_loss = str_loss + tmp_loss_str + '\n'
        
    # str_auc = ' '.join(['AUC{}: {:.4f}'.format(i+1, test_auc[i]) for i in range(num_class)])
    # str_loss = ' '.join(['Average loss{}: {:.4f}'.format(i+1, test_loss_list[i]) for i in range(num_class)])

    logger.info('Test set: Average loss: {:.4f} \n {} \n {}'.format(test_loss, str_loss, str_auc))
    
    # if test_auc[3] > best_auc: # and test_auc[0] > 0.81 and test_auc[1] > 0.83:
    #     best_auc = test_auc[1]
    #     torch.save(model.state_dict(), './modelSaved/mfh_v1_0.pth')
    #     model1 = torch.jit.script(model)
    #     torch.jit.save(model1, './modelSaved/mfh_v1_0.pt')
    #     logger.info('Save model with loss: {:.4f}, {}, {}'.format(test_loss, str_loss, str_auc))
    
    if epoch % 1 == 0:
        torch.save(model.state_dict(), './modelSaved/mfh_v1_2_epoch{}.pth'.format(str(epoch+1)))
        model1 = torch.jit.script(model)
        torch.jit.save(model1, './modelSaved/mfh_v1_2_epoch{}.pt'.format(str(epoch+1)))
        # logger.info('Save model with loss: {:.4f}, {}, {}, epoch: {}'.format(test_loss, str_loss, str_auc, str(epoch+1)))
        logger.info('Save model epoch: {}'.format(str(epoch+1)))


if __name__ == '__main__':
    # get configuration
    __CONF_PATH = './config/mfh.yaml'
    config = util.get_config(__CONF_PATH)
    util.seed_everything(config.Seed)

    logger.info(config.Model)
    logger.info(config.Train)

    # get data
    train_loader = DataLoader(MFHNpDatasetLoader(config.Data.filePath, 'train'), config.Train.batchSize, shuffle=True, num_workers=config.Train.numWorkers)
    eval_loader = DataLoader(MFHNpDatasetLoader(config.Data.filePath, 'val'), config.Train.batchSize, shuffle=False, num_workers=config.Train.numWorkers)
    
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
    
    model = MFHATT(config, feature_columns).to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):  #(nn.Conv2d, nn.Linear)
            nn.init.xavier_uniform_(m.weight)
            # nn.init.kaiming_uniform_(m.weight)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Train.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.Train.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.Train.milestones, gamma=config.Train.gamma)
    train(train_loader, eval_loader, model, optimizer, scheduler, config.Train.epochs, config.Model['mode'], config.Train['stage'])




