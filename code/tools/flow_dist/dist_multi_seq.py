# -*- coding: utf-8 -*-
import pandas as pd
import pulp
import math
import numpy as np
import hashlib
import warnings

import utils

from joblib import Parallel, delayed

warnings.filterwarnings('ignore')


def cal_each_LP(group_index, df_group, 
                list_cols_score, list_item, list_ratio_item, 
                coef_flow_shrink, 
                group_num, 
                is_lp_file=0, lp_file_prefix='', 
                is_rule=0, list_cols_rule=[]):
    '''
    LP each group
    '''
    df_group.reset_index(drop=True, inplace=True)

    rows = df_group.shape[0]
    cols = len(list_cols_score)

    # 构造函数
    func = pulp.LpProblem('item_dist', sense=pulp.LpMaximize)

    # 决策变量
    var_choices = pulp.LpVariable.dicts('Choices', (range(rows), range(cols)), lowBound=0, upBound=1, cat=pulp.LpBinary)

    # s.t. 1：规则限定，隐式写在决策变量中
    if is_rule == 1:
        for i, col in enumerate(list_cols_rule):
            index = df_group[df_group[col]==0].index.values
            for j in index:
                var_choices[j][i].upBound = 0

    # obj
    func += pulp.lpSum([var_choices[i][j] * df_group.loc[i, col]
                        for i in range(rows)
                        for j, col in enumerate(list_cols_score)]), 'maximize_score'

    # s.t. 2：每组只有一个item被选择
    for i in range(rows):
        func += pulp.lpSum([var_choices[i][j] for j in range(cols)]) == 1, ''
        
    # s.t. 3：各item流量占比
    for j in range(cols):
        func += pulp.lpSum([var_choices[i][j] for i in range(rows)]) <= math.ceil(
            rows * list_ratio_item[j] * coef_flow_shrink), 'flow {}'.format(list_item[j])

    # 保存lp文件
    if is_lp_file == 1:
        func.writeLP(lp_file_prefix+'_{}_{}.lp'.format(group_num, str(group_index).zfill(len(str(group_num)))))

    solver = pulp.PULP_CBC_CMD(msg=False)
    func.solve(solver)

    if pulp.LpStatus[func.status] != 'Optimal':
        raise Exception('group {} slove wrong!!!'.format(group_index))

    df_group['item'] = ''
    for i in range(rows):
        for j in range(cols):
            if pulp.value(var_choices[i][j]) == 1:
                df_group.loc[i, 'item'] = list_item[j]
                continue

    return df_group


class dist_multi_LP(object):
    def __init__(self, 
                 cols_score, cols_score_dist, item, ratio_item, 
                 coef_flow_shrink, 
                 cols_rep, 
                 group_salt, group_num, worker_num, 
                 is_lp_file=0, lp_file_prefix='', 
                 is_rule=0, cols_rule=[], cols_rule_dist=[]):
        self.list_cols_score = cols_score
        self.list_cols_score_dist = cols_score_dist
        self.list_item = item
        self.list_ratio_item = ratio_item
        self.coef_flow_shrink = coef_flow_shrink
        self.dict_cols_rep = cols_rep
        self.group_salt = group_salt
        self.group_num = group_num
        self.worker_num = worker_num
        self.is_lp_file = is_lp_file
        self.lp_file_prefix = lp_file_prefix
        self.is_rule = is_rule
        self.list_cols_rule = cols_rule
        self.list_cols_rule_dist = cols_rule_dist
        
    
    def __apply_parallel(self, df_grouped, func, para={}, n_jobs=8):
        '''
        multi process
        '''
        list_paraller = Parallel(n_jobs=n_jobs)(delayed(func)(name, df_group, **para) for name, df_group in df_grouped)
        
        return list_paraller
    
    
    def LP(self, df):
        # 打分排序
        df['score_list'] = df.apply(lambda x: [str(round(s, 10)) for s in sorted(np.array([x[col] for col in self.list_cols_score_dist]), reverse=True)], axis=1)
        
        if len(self.list_cols_score) != len(self.list_item):
            raise Exception('item wrong!!!')
            
        list_item_dist = [self.list_item[i] for i, x in enumerate(self.list_cols_score) if x in self.list_cols_score_dist]
        df['item_list'] = df.apply(lambda x: [list_item_dist[i] for i in np.argsort(np.array([x[col] for col in self.list_cols_score_dist]))[::-1]], axis=1)
        
        # 任意规则均不满足的数据直接输出排序
        df_none = pd.DataFrame({'uid': [], 'item_list': [], 'score_list': []})
        if self.is_rule == 1:
            if len(self.list_cols_rule) != len(self.list_cols_score):
                raise Exception('rules wrong!!!')
                
            df['rule_none'] = df[self.list_cols_rule_dist].max(axis=1)
            
            df_none = df[df['rule_none']==0][['uid', 'item_list', 'score_list']]
            df_none.reset_index(drop=True, inplace=True)
            
            df = df[(df['rule_none']==1)|df['rule_none'].isna()]
            df.reset_index(drop=True, inplace=True)
            
        # 人群分组
        df['group'] = df['uid'].apply(lambda x: utils.md5_hash(hashlib.sha1((str(x)+self.group_salt).encode('utf-8')).hexdigest())%self.group_num)
        
        # 局部线性规划，多进程并行
        df_grouped = df.groupby(by=['group'])
        
        dict_para = {
            'list_cols_score': self.list_cols_score, 
            'list_item': self.list_item, 
            'list_ratio_item': self.list_ratio_item, 
            'coef_flow_shrink': self.coef_flow_shrink, 
            'group_num': self.group_num, 
            'is_lp_file': self.is_lp_file, 
            'lp_file_prefix': self.lp_file_prefix, 
            'is_rule': self.is_rule, 
            'list_cols_rule': self.list_cols_rule
        }
        list_df_grouped_LP = self.__apply_parallel(df_grouped, cal_each_LP, para=dict_para, n_jobs=self.worker_num)
        
        # 合并局部结果
        df_grouped_LP = pd.concat(list_df_grouped_LP, axis=0)
        df_grouped_LP.reset_index(drop=True, inplace=True)
        
        # 分配结果修改首位排序
        df_grouped_LP['pos'] = df_grouped_LP.apply(lambda x: x['item_list'].index(x['item']), axis=1)
        
        list_order = list(range(len(list_item_dist)))
        df_grouped_LP['score_order'] = df_grouped_LP.apply(lambda x: '-'.join([x['score_list'][j] for j in [x['pos']] + [i for i in list_order if i != x['pos']]]), axis=1)
        df_grouped_LP['item_order'] = df_grouped_LP.apply(lambda x: '-'.join([x['item_list'][j] for j in [x['pos']] + [i for i in list_order if i != x['pos']]]), axis=1)
        
        # 合并全部结果
        df_none['score_order'] = df_none['score_list'].apply(lambda x: '-'.join(x))
        df_none['item_order'] = df_none['item_list'].apply(lambda x: '-'.join(x))
        
        df_res = pd.concat([df_grouped_LP[['uid', 'item_order', 'score_order']], 
                            df_none[['uid', 'item_order', 'score_order']]], axis=0)
        df_res.reset_index(drop=True, inplace=True)
        df_res.rename(columns={'item_order': self.dict_cols_rep['item_order'], 
                               'score_order': self.dict_cols_rep['score_order']}, 
                      inplace=True)
        
        return df_res
    
    
if __name__ == '__main__':
    ## config
    # base
    cols_score = ['score_01', 'score_02', 'score_03', 'score_04', 'score_10']
    cols_score_dist = ['score_03', 'score_04']
    item = ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
    ratio_item = [0.0, 0.0, 0.3, 0.7, 0.0]
    coef_flow_shrink = 1.1
    cols_rep = {
        'item_order': 'order', 
        'score_order': 'score_order'
    }
    res_file = 'flow_dist/data/df_lp_res.pickle'
    
    # rule
    is_rule = 1
    cols_rule = ['rule_01', 'rule_02', 'rule_04', 'rule_06', 'rule_03']
    cols_rule_dist = ['rule_04', 'rule_06']
    
    # multi_proc
    group_salt = '202308'
    group_num = 100
    worker_num = 10
    is_lp_file = 1
    lp_file_prefix = 'flow_dist/data/multi_proc/flow'
    
    ## data read
    df_rule_score = utils.load_data('flow_dist/data/ugrs.txt')
    
    ## LP
    dml = dist_multi_LP(cols_score, cols_score_dist, item, ratio_item, 
                        coef_flow_shrink, 
                        cols_rep, 
                        group_salt, group_num, worker_num, 
                        is_lp_file=is_lp_file, lp_file_prefix=lp_file_prefix, 
                        is_rule=is_rule, cols_rule=cols_rule, cols_rule_dist=cols_rule_dist)
    df_res = dml.LP(df_rule_score)
    
    ## data save
    utils.save_pickle(df_res, res_file)
    