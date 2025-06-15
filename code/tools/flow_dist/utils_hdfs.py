# -*- coding: utf-8 -*-
import pandas as pd
import time
import requests
import json
import datetime
import os
import pickle

from dynaconf import Dynaconf


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


def sendPost(data, url='https://'):
    headers = {'Content-type': 'application/json'}
    robot_url = url
    text = json.dumps(data)
    response = requests.post(url=robot_url, headers=headers, data=text)


def chat(flag, title='', content='', url='https://'):
    '''
    chat message
    '''
    text = '{}流量分发结果：\n'.format(title)
    if flag == 1:
        text += '{}无产出 \n**发生错误，请及时检查!** \n'.format(content)
    else:
        text += content
        text += '\n'

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text += '检查时间: ' + now
    postdata = {
        'msgtype': 'markdown',
        'text': '{}流量分发监控 @jarretthan '.format(title),  # @group 所有人 @jarretthan
        # 'at': {'isAtAll': True},
        'attachments': [
            {
                'text': text,
                'color': '#ffa500'
            }
        ]
    }
    sendPost(postdata, url=url)


def hive_data(dict_hdfs,
              dt, group, sql_file, output_file,
              sleep_once_time=1800, check_cnts=48,
              yarn_queue='xxx',
              para_chat={'is_chat': 1,
                          'title': '', 'url': 'https://'}):
    '''
    get data from hive table
    '''

    # check if the tables exists
    list_check_table = [k for k, _ in dict_hdfs.items()]
    list_check_cmd = ['hadoop fs -test -e {}'.format(v) for _, v in dict_hdfs.items()]
    list_flag = [1 for _ in list_check_cmd]
    index = 0

    while True:
        if index > check_cnts:
            str_res = '\n{} hive tables not found!!!'.format(sum([1 for x in list_flag if x != 0]))
            for i in range(len(list_flag)):
                if list_flag[i] != 0:
                    str_res += '\n' + list_check_table[i]
            if para_chat['is_chat'] == 1:
                chat(1, title=para_chat['title'], content=str_res, url=para_chat['url'])
            raise Exception(str_res)
            
        if index == int(check_cnts/4):
            chat(0, title=para_chat['title'], content='\nhive tables check pass 1/4 time!!!', url=para_chat['url'])

        list_flag = [os.system(x) for x in list_check_cmd]

        if max(list_flag) == 0:
            print('tables exists')
            break
        else:
            for i in range(len(list_flag)):
                if list_flag[i] != 0:
                    print('hive table {} not found!!!'.format(list_check_table[i]))
            print('waiting for hive table to be created..., {}/{}.'.format(index + 1, check_cnts))
            time.sleep(sleep_once_time)
            index += 1

    # get hive data
    cmd = '''hive -hiveconf mapreduce.job.queuename={} -hiveconf dt={} -hiveconf g={} -f {} > {}'''.format(
        yarn_queue, dt, group, sql_file, output_file)
    print(cmd)
    signal = os.system(cmd)
    if signal != 0:
        print('hive command failed: {}!!!'.format(cmd))
        if para_chat['is_chat'] == 1:
            chat(1, title=para_chat['title'], content='\nhive command failed: {}!!!'.format(cmd), url=para_chat['url'])
        raise Exception('hive command failed!!!')

    return load_data(output_file)


def hdfs_data(dict_hdfs,
              hdfs_path, data_dir,
              sleep_once_time=1800, check_cnts=48,
              para_chat={'is_chat': 1,
                          'title': '', 'url': 'https://'}):
    '''
    get data from hdfs
    '''

    # check if the hdfs data exists
    list_check_table = [k for k, _ in dict_hdfs.items()]
    list_check_cmd = ['hadoop fs -test -e {}'.format(v) for _, v in dict_hdfs.items()]
    list_flag = [1 for _ in list_check_cmd]
    index = 0

    while True:
        if index > check_cnts:
            str_res = '\n{} hdfs data not found!!!'.format(sum([1 for x in list_flag if x != 0]))
            for i in range(len(list_flag)):
                if list_flag[i] != 0:
                    str_res += '\n' + list_check_table[i]
            if para_chat['is_chat'] == 1:
                chat(1, title=para_chat['title'], content=str_res, url=para_chat['url'])
            raise Exception(str_res)
            
        if index == int(check_cnts/4):
            chat(0, title=para_chat['title'], content='\nhdfs data check pass 1/4 time!!!', url=para_chat['url'])

        list_flag = [os.system(x) for x in list_check_cmd]

        if max(list_flag) == 0:
            print('hdfs data exists')
            print('creating...wait for 30 mins...')
            time.sleep(sleep_once_time)
            break
        else:
            for i in range(len(list_flag)):
                if list_flag[i] != 0:
                    print('hdfs data {} not found!!!'.format(list_check_table[i]))
            print('waiting for hdfs data to be created..., {}/{}.'.format(index + 1, check_cnts))
            time.sleep(sleep_once_time)
            index += 1

    # get hdfs data
    cmd_clean = '''rm -rf {}'''.format(data_dir)
    os.system(cmd_clean)
    
    cmd = '''hadoop fs -get {} {}'''.format(hdfs_path, data_dir)
    print(cmd)
    signal = os.system(cmd)
    if signal != 0:
        print('hdfs command failed: {}!!!'.format(cmd))
        if para_chat['is_chat'] == 1:
            chat(1, title=para_chat['title'], content='\nhive command failed: {}!!!'.format(cmd), url=para_chat['url'])
        raise Exception('hdfs command failed!!!')

    return load_data(os.path.join(data_dir, os.listdir(data_dir)[0]))


def quant_cut(df, column, n):
    '''
    :param df: dataframe of the original data
    :param column: column name
    :param n: quantile number
    :param dic: dictionary to store quantile
    :return:
    '''
    dic = {}
    for i in range(n, -1, -1):
        try:
            tmp = pd.qcut(df[column], q=n, labels=list(range(i)), duplicates='drop')
            df[column + '_quant'] = tmp
            dic[column + '_quant'] = i
            break
        except:
            continue

    if dic[column + '_quant'] == 0:
        df[column + '_quant'] = 0


def load_data(file):
    '''
    :param file: file path
    :return: dataframe of the data
    '''
    df = pd.read_csv(file, sep='\t', encoding='utf-8')

    return df


def statistics(df, date, role, exp_mode,
               list_index_exp_group, item_name, list_item, list_code_item,
               list_ratio_item, coef_flow_shrink,
               para_chat={'is_chat': 1,
                           'title': '',
                           'url': 'https://'}):
    message = '\n数据日期：{}'.format(date) + \
              '\n角色：{}'.format(role) + \
              '\n实验类型：{}'.format(exp_mode) + \
              '\n实验组人群分组索引：{}'.format(list_index_exp_group) + \
              '\n实验组{}分组：{}'.format(item_name, list_item) + \
              '\n实验组{}分组线上编码：{}'.format(item_name, list_code_item) + \
              '\n实验组总流量：{}'.format(df.shape[0]) + \
              '\n实验组{}分组流量占比（预期）：{}'.format(item_name, list_ratio_item) + \
              '\n实验组{}分组流量缩放比例：{}'.format(item_name, coef_flow_shrink) + \
              '\n实验组{}分组流量（实际）：{}'.format(item_name, [df[df['xxx_code']==x].shape[0] for x in list_code_item]) + \
              '\n实验组{}分组流量占比（实际）：{}'.format(item_name, [round(df[df['xxx_code']==x].shape[0]/df.shape[0], 3)
                                                                 for x in list_code_item])
    print(message)
    if para_chat['is_chat'] == 1:
        chat(0, title=para_chat['title'], content=message, url=para_chat['url'])


def save_pickle(data, file_path_name):
    with open(file_path_name, 'wb') as f:
        pickle.dump(data, f, protocol=4)
