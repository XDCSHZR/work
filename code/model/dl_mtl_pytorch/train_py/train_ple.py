import numpy as np
import utils.utils as util
import torch
from utils.dataset import *
from model.ple import *
from loss.focalloss import FocalLossV1 as FocalLoss
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = util.get_logger('')
best_loss = np.inf
best_auc = 0


def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}


def denseFeature(feat):
    return {'feat': feat}


@util.timer
def data_preprocess(config):
    df = pd.read_csv(config.Data['trainFilePath'], sep='\t')
    app_cols = ['xxx']
    df.pop('uid')
    df.pop('xxx')
    df.pop('xxx')
    df.pop('xxx')
    df.pop('xxx')
    for i in app_cols:
        df.pop(i)
    y = df['label']
    x = df[df.columns[:-1]]
    print(x.shape, y.shape)
    # print(df['label'].value_counts())
    print('data preprocessing ...')
    categorical_columns = ['xxx']

    categorical_dims = {'xxx': 20}

    x[categorical_columns] = x[categorical_columns].where(x[categorical_columns] >= 0, 19)
    x[categorical_columns] = x[categorical_columns].where(x[categorical_columns] < 19, 19)
    continous_columns = [i for i in x.columns if i not in categorical_columns]
    x = pd.concat([x[categorical_columns], x[continous_columns]], axis=1)
    feature_columns = [[denseFeature(feat) for feat in continous_columns]] + \
                      [[sparseFeature(feat, categorical_dims[feat], config.Model['embed_dim']) for feat in categorical_columns]]

    # y = OneHotEncoder().fit_transform(y.values.reshape(-1, 1)).toarray()
    if config.Model.mode == 'multi-label':  # multi-label or multi-class
        df['label_1'] = df['label'].replace(2, 1)  # label > 0 => 1
        df['label_2'] = df['label'].replace(1, 0)  # label < 2 => 0
        df['label_2'] = df['label_2'].replace(2, 1)
        print(df['label_1'].value_counts(), df['label_2'].value_counts())
        y = df[['label_1', 'label_2']]
    
    print(x.shape)
    return x.values, y.values, feature_columns


def train(train_loader, test_loader, model, optimizer, scheduler, criterion, epochs, data_test, mode):
    for epoch in range(epochs):
        model.train()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device).to(torch.float32), y.to(device).to(torch.float32)
            optimizer.zero_grad()
            output = model(x)
            loss0, loss1 = criterion[0](output[:, 0], y[:, 0]), criterion[1](output[:, 1], y[:, 1])
            loss = criterion[1](output[:, 1], y[:, 1]) + criterion[0](output[:, 0], y[:, 0])
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                logger.info('Epoch: [{}/{}], Step: [{}/{}], Lr: {:.6f}, Loss0: {:.6f}, Loss1: {:.6f}, Loss: {:.6f}'.format(
                    epoch + 1, epochs, i + 1, len(train_loader), optimizer.param_groups[0]['lr'], loss0.item(), loss1.item(), loss.item()))
            if i % 5000 == 0:
                val(test_loader, model, criterion, data_test, mode)

        scheduler.step()
        

def val(test_loader, model, criterion, data_test, mode):
    model.eval()
    global best_loss
    global best_auc
    test_loss = 0
    test_loss0 = 0
    test_loss1 = 0
    y_pred = []

    y_true = data_test[1]
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device).to(torch.float32), y.to(device).to(torch.float32)
            output = model(x)
            output1 = torch.sigmoid(output)
            test_loss0 = test_loss0 + criterion[0](output[:,0], y[:,0]).item()
            test_loss1 = test_loss1 + criterion[1](output[:,1], y[:,1]).item()
            test_loss = test_loss + criterion[0](output[:,0], y[:,0]).item() + criterion[1](output[:,1], y[:,1]).item()
            if i == 0:
                y_pred = output1.cpu().numpy()
            else:
                y_pred = np.concatenate((y_pred, output1.cpu().numpy()), axis=0)
    
    test_loss0 /= len(test_loader)
    test_loss1 /= len(test_loader)
    test_loss /= len(test_loader)
    
    num_class = y_true.shape[1]
    # print(num_class, y_true.shape, y_pred.shape)
    test_auc = [util.auc(y_true[:, i], y_pred[:, i]) for i in range(num_class)]
    str_auc = ' '.join(['AUC{}: {:.4f}'.format(i, test_auc[i]) for i in range(num_class)])
    logger.info('Test set: Average loss0: {:.4f}, Average loss1: {:.4f}, Average loss: {:.4f}, {}'.format(test_loss0, test_loss1, test_loss, str_auc))
    # logger.info('Test set: Average loss: {:.4f}, AUC: {:.4f}\n'.format(test_loss, test_auc))

    # using best loss to save model
    # if test_loss < best_loss:
    #     best_loss = test_loss
    #     torch.save(model.state_dict(), './modelSaved/model.pth')
    #     model1 = torch.jit.script(model)
    #     torch.jit.save(model1, './modelSaved/mmoe.pt')
    #     logger.info('Save model with loss: {:.4f}, {}'.format(test_loss, str_auc))
    
    if test_auc[-1] > best_auc:
        best_auc = test_auc[-1]
        torch.save(model.state_dict(), './modelSaved/model.pth')
        model1 = torch.jit.script(model)
        torch.jit.save(model1, './modelSaved/ple.pt')
        logger.info('Save model with loss: {:.4f}, {}'.format(test_loss, str_auc))


if __name__ == '__main__':
    # get configuration
    __CONF_PATH = './config/ple.yaml'
    config = util.get_config(__CONF_PATH)
    util.seed_everything(config.Seed)

    logger.info("Model: PLE")
    logger.info(config.Model)
    logger.info(config.Train)

    # get data
    x, y, feature_columns = data_preprocess(config)
    X_train, X_test, y_train, y_test = util.split_data(x, y, ratio=0.2, shuffle=True)
    train_loader = DataLoader(DatasetLoader(X_train, y_train), config.Train.batchSize, shuffle=True, num_workers=config.Train.numWorkers)
    eval_loader = DataLoader(DatasetLoader(X_test, y_test), config.Train.batchSize, shuffle=False, num_workers=config.Train.numWorkers)
    

    model = MLPPLE(config, feature_columns).to(device)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):  #(nn.Conv2d, nn.Linear)
            # nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


    if config.Model['loss'] == 'focalloss':
        criterion = (nn.BCEWithLogitsLoss().to(device), FocalLoss(multi_class=False).to(device))
    else:
        criterion = (nn.BCEWithLogitsLoss().to(device), nn.BCEWithLogitsLoss().to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Train.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.Train.milestones, gamma=config.Train.gamma)
    train(train_loader, eval_loader, model, optimizer, scheduler, criterion, config.Train.epochs, (X_test, y_test), config.Model['mode'])

