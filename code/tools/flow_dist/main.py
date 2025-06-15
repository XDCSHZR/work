# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd

import utils
import dist

from conf import Configuration
from dist_multi import dist_multi_LP


if __name__ == '__main__':
    config = Configuration()

    print('data')
    df_rule_score = utils.hive_data(config.conf['check_conf']['hdfs'],
                                    config.date, "'("+str(config.conf['dist_conf']['base']['index_exp_group'])[1:-1]+")'", config.conf['down_conf']['sql_file'], config.conf['down_conf']['data_file'],
                                    sleep_once_time=config.conf['check_conf']['sleep_once_time'], check_cnts=config.conf['check_conf']['check_cnts'],
                                    yarn_queue=config.conf['down_conf']['yarn_queue'],
                                    para_chat=config.conf['chat_conf'])

    print('LP')
    if config.conf['dist_conf']['multi_proc']['is_multi'] == 1:
        print('choose multi process')
        dml = dist_multi_LP(config.conf['dist_conf']['base']['cols_score'], config.conf['dist_conf']['base']['item'], config.conf['dist_conf']['base']['code_item'], config.conf['dist_conf']['base']['ratio_item'], 
                            config.conf['dist_conf']['base']['coef_flow_shrink'], 
                            config.conf['dist_conf']['base']['cols_rep'], 
                            config.conf['dist_conf']['multi_proc']['group_salt'], config.conf['dist_conf']['multi_proc']['group_num'], config.conf['dist_conf']['multi_proc']['worker_num'], 
                            is_lp_file=config.conf['dist_conf']['multi_proc']['is_lp_file'], lp_file_prefix=config.conf['dist_conf']['multi_proc']['lp_file_prefix'], 
                            is_rule=config.conf['dist_conf']['rule']['is_rule'], cols_rule=config.conf['dist_conf']['rule']['cols_rule'], cols_rule_dist=config.conf['dist_conf']['rule']['cols_rule_dist'])
        df_res = dml.LP(df_rule_score)
    else:
        print('choose group simplify')
        df_res = dist.LP(df_rule_score, 
                         config.conf['dist_conf']['base']['cols_score'], config.conf['dist_conf']['base']['item'], config.conf['dist_conf']['base']['code_item'], config.conf['dist_conf']['base']['ratio_item'], 
                         config.conf['dist_conf']['base']['coef_flow_shrink'], 
                         config.conf['dist_conf']['base']['cols_rep'], 
                         config.conf['dist_conf']['group_simp']['cols_score_group'], 
                         config.conf['dist_conf']['group_simp']['qcut_num'], 
                         config.conf['dist_conf']['group_simp']['lp_file'], 
                         is_rule=config.conf['dist_conf']['rule']['is_rule'], cols_rule=config.conf['dist_conf']['rule']['cols_rule'], cols_rule_dist=config.conf['dist_conf']['rule']['cols_rule_dist'], coef_score_rule=config.conf['dist_conf']['group_simp']['coef_score_rule'])
        
    utils.save_pickle(df_res, config.conf['dist_conf']['base']['res_file'])
    
    print('statistics')
    utils.statistics(df_res, config.date, config.conf['stat_conf']['role'], config.conf['stat_conf']['exp_mode'],
                     config.conf['dist_conf']['base']['index_exp_group'], config.conf['stat_conf']['item_name'], config.conf['dist_conf']['base']['item'], config.conf['dist_conf']['base']['code_item'],
                     config.conf['dist_conf']['base']['ratio_item'], config.conf['dist_conf']['base']['coef_flow_shrink'],
                     para_chat=config.conf['chat_conf'], 
                     item_none_name=config.conf['stat_conf']['item_none_name'], item_none_code=config.conf['stat_conf']['item_none_code'])
    
    print('upload')
    df_res.to_csv(config.conf['upload_conf']['data_file'], sep='\t', encoding='utf-8', index=False, header=False)

    cmd = '''hive -hiveconf upload_file={} -hiveconf yesterday_year={} -hiveconf yesterday_month={} -hiveconf yesterday_day={} -f {}'''.format(
        config.conf['upload_conf']['data_file'], config.year, config.month, config.day, config.conf['upload_conf']['sql_file'])
    signal = os.system(cmd)
    if signal == 0:
        print('upload result to hive successfully!')
    else:
        print('upload result to hive failed!!!')
        if config.conf['chat_conf']['is_chat'] == 1:
            utils.chat(1, title=config.conf['chat_conf']['title'], content='\nupload result to hive failed!!!', url=config.conf['chat_conf']['url'])
        raise Exception('upload result to hive failed!!!')
