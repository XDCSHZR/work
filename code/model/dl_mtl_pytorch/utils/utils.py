import os
from dynaconf import Dynaconf
from loguru import logger
import time
import torch
import torch.nn as nn
import random
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split


def get_config(conf_path):
    """
    get config from configuration direction, supported file formats: yaml
    """
    if not os.path.exists(conf_path):
        raise Exception('config file not found')

    if os.path.isfile(conf_path):
        conf_list = [conf_path]
    else:
        conf_list = []
        for root, _, files in os.walk(conf_path):
            for file in files:
                if os.path.splitext(file)[1] == '.yaml':  # yaml files
                    conf_list.append(os.path.join(root, file))

    if not conf_list:
        raise Exception("config file not found")

    settings = Dynaconf(
        envvar_prefix="DYNACONF",
        settings_files=conf_list,
    )
    return settings


def get_logger(config, algorithm='log'):
    """
    get logger from config
    """
    try:
        log_path = config.log['path'] if 'path' in config.log else None
        rotation = config.log['rotation'] if 'rotation' in config.log else None
        compression = config.log['compression'] if 'compression' in config.log else None
        retention = config.log['retention'] if 'retention' in config.log else 1000
        serialize = config.log['serialize'] if 'serialize' in config.log else False
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        log_file = os.path.join(log_path, algorithm + "_{time}.log")
        logger.add(log_file, rotation=rotation, compression=compression, retention=retention,
                   serialize=serialize)
    except AttributeError:
        __DEFAULT_LOG_PATH = 'logs/'  # 不配置，使用默认路径
        if not os.path.exists(__DEFAULT_LOG_PATH):
            os.mkdir(__DEFAULT_LOG_PATH)
        log_file = os.path.join(__DEFAULT_LOG_PATH, algorithm + "_{time}.log")
        logger.add(log_file)
    return logger


def timer(myFunction):
    """
    caculate function time cost.
    """
    name = myFunction.__name__

    def wrapper(*args, **kwargs):
        start = time.time()
        result = myFunction(*args, **kwargs)
        end = time.time()
        print('Function {} cost time: {:.5f} s'.format(name, end - start))
        return result

    return wrapper


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred)


def split_data(x, y, ratio, shuffle):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio,
                                                        stratify=y,
                                                        shuffle=shuffle)
    return x_train, x_test, y_train, y_test


def split_data_stratify(x, y, ratio, shuffle, stratify):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio,
                                                        stratify=stratify,
                                                        shuffle=shuffle)
    return x_train, x_test, y_train, y_test


def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}


def denseFeature(feat):
    return {'feat': feat}