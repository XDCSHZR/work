# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd

import utils
import dist

from conf import Configuration


if __name__ == '__main__':
    config = Configuration()

    print('data')
    df_rule_score = utils.hdfs_data(config.conf['check_conf']['hdfs'],
                                    config.conf['down_conf']['hdfs_path'], config.conf['down_conf']['data_dir'],
                                    sleep_once_time=config.conf['check_conf']['sleep_once_time'], check_cnts=config.conf['check_conf']['check_cnts'],
                                    para_chat=config.conf['chat_conf'])

    print('LP')
    df_res = dist.LP(df_rule_score, config.conf['dist_conf']['cols_score'],
                     config.conf['dist_conf']['cols_score_group'],
                     config.conf['dist_conf']['item'], config.conf['dist_conf']['ratio_item'], config.conf['dist_conf']['coef_flow_shrink'],
                     config.conf['dist_conf']['lp_file'],
                     config.conf['dist_conf']['code_item'],
                     config.conf['dist_conf']['cols_rep'],
                     qcut_num=config.conf['dist_conf']['qcut_num'], is_rule=config.conf['dist_conf']['is_rule'], cols_rule=config.conf['dist_conf']['cols_rule'], coef_score_rule=config.conf['dist_conf']['coef_score_rule'])
    utils.save_pickle(df_res, config.conf['dist_conf']['res_file'])

    print('statistics')
    utils.statistics(df_res, config.date, config.conf['stat_conf']['role'], config.conf['stat_conf']['exp_mode'],
                     config.conf['dist_conf']['index_exp_group'], config.conf['stat_conf']['item_name'], config.conf['dist_conf']['item'], config.conf['dist_conf']['code_item'],
                     config.conf['dist_conf']['ratio_item'], config.conf['dist_conf']['coef_flow_shrink'],
                     para_chat=config.conf['chat_conf'])

    print('upload')
    df_res.to_csv(config.conf['upload_conf']['data_file'], sep='\t', encoding='utf-8', index=False, header=False)

    cmd = '''spark-sql --queue {} -d upload_file={} -d yesterday_year={} -d yesterday_month={} -d yesterday_day={} -f {}'''.format(
        config.conf['down_conf']['yarn_queue'], config.conf['upload_conf']['data_file'], config.year, config.month, config.day, config.conf['upload_conf']['sql_file'])
    signal = os.system(cmd)
    if signal == 0:
        print('upload result to hive successfully!')
    else:
        print('upload result to hive failed!!!')
        if config.conf['chat_conf']['is_chat'] == 1:
            utils.chat(1, title=config.conf['chat_conf']['title'], content='\nupload result to hive failed!!!', url=config.conf['chat_conf']['url'])
        raise Exception('upload result to hive failed!!!')
