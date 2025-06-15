# -*- coding: utf-8 -*-
import pandas as pd
import pulp
import math
import numpy as np

import utils

from functools import reduce


def LP(df, 
       cols_score, cols_score_dist, item, ratio_item, 
       coef_flow_shrink, 
       cols_rep, 
       cols_score_group, 
       qcut_num, 
       lp_file, 
       is_rule=0, cols_rule=[], cols_rule_dist=[], coef_score_rule=1e-10):
    # 打分排序
    df['score_list'] = df.apply(lambda x: [str(round(s, 10)) for s in sorted(np.array([x[col] for col in cols_score_dist]), reverse=True)], axis=1)
    
    if len(cols_score) != len(item):
        raise Exception('item wrong!!!')
        
    item_dist = [item[i] for i, x in enumerate(cols_score) if x in cols_score_dist]
    df['item_list'] = df.apply(lambda x: [item_dist[i] for i in np.argsort(np.array([x[col] for col in cols_score_dist]))[::-1]], axis=1)
    
    # 对任意规则均不满足的数据直接输出排序
    df_none = pd.DataFrame({'uid': [], 'item_list': [], 'score_list': []})
    cols_score_rule = []
    cols_score_group_rule = []
    if is_rule == 1:
        if len(cols_rule) != len(cols_score):
            raise Exception('rules wrong!!!')
            
        df['rule_none'] = df[cols_rule_dist].max(axis=1)
        
        df_none = df[df['rule_none']==0][['uid', 'item_list', 'score_list']]
        df_none.reset_index(drop=True, inplace=True)
        
        # s.t. 1：规则限定，分数降权
        df = df[(df['rule_none']==1)|df['rule_none'].isna()]
        df.reset_index(drop=True, inplace=True)

        cols_score_rule = [x + '_rule' for x in cols_score]
        cols_score_group_rule = [x + '_rule' for x in cols_score_group]

        for i in range(len(cols_rule)):
            df[cols_score_rule[i]] = df[cols_score[i]] * (1-(1-df[cols_rule[i]])*(1-coef_score_rule))
    else:
        cols_score_rule = cols_score
        cols_score_group_rule = cols_score_group

    # 排序分桶
    if len(qcut_num) != len(cols_score_group_rule):
        raise Exception('qcut num wrong!!!')
        
    for i, col in enumerate(cols_score_group_rule):
        utils.quant_cut(df, col, qcut_num[i])
        
    cols_score_group_rule_quant = [x + '_quant' for x in cols_score_group_rule]
    
    qcut_num_carry_int = [1] * len(qcut_num)
    qcut_num_copy = qcut_num
    for i, qcn in enumerate(qcut_num_copy):
        while qcn > 1:
            qcn /= 10
            qcut_num_carry_int[i] *= 10

    df['group'] = reduce(lambda x, y: x*y, qcut_num_carry_int)
    qcut_num_carry_int_each_step = 1
    for i, col in enumerate(cols_score_group_rule_quant[::-1]):
        if i > 0:
            qcut_num_carry_int_each_step *= qcut_num_carry_int[::-1][i-1]
        df['group'] += df[col].astype(np.int32) * qcut_num_carry_int_each_step

    # 分桶后最优化
    df_group = df[['group', 'uid']].groupby(by='group', as_index=False).count().\
        merge(df[['group']+cols_score_rule].groupby(by='group', as_index=False).mean(), on='group', how='left')
    df_group.rename(columns={'uid': 'count'}, inplace=True)
    df_group.rename(columns={x: x + '_mean' for x in cols_score_rule}, inplace=True)

    cols_score_rule_mean = [x + '_mean' for x in cols_score_rule]

    rows = df_group.shape[0]
    cols = len(cols_score_rule_mean)

    # 构造函数
    func = pulp.LpProblem('item dist', sense=pulp.LpMaximize)

    # 决策变量
    var_choices = pulp.LpVariable.dicts('Choices', (range(rows), range(cols)), lowBound=0, upBound=1, cat=pulp.LpBinary)

    # obj
    func += pulp.lpSum([var_choices[i][j] * df_group.iloc[i, j+2]
                        for i in range(rows)
                        for j in range(cols)]), 'maximize score'

    # s.t. 2：每组只有一个item被选择
    for i in range(rows):
        func += pulp.lpSum([var_choices[i][j] for j in range(cols)]) == 1, ''

    # s.t. 3：各item流量占比
    for j in range(cols):
        func += pulp.lpSum([var_choices[i][j] * df_group.loc[i, 'count'] for i in range(rows)]) <= math.ceil(
            df_group['count'].sum() * ratio_item[j]) * coef_flow_shrink, 'flow {}'.format(item[j])

    # 保存lp文件
    func.writeLP(lp_file)

    # 求解
    solver = pulp.PULP_CBC_CMD(msg=False)
    func.solve(solver)
    
    if pulp.LpStatus[func.status] != 'Optimal':
        raise Exception('slove wrong!!!')

    df_group['item'] = ''
    for i in range(rows):
        for j in range(cols):
            if pulp.value(var_choices[i][j]) == 1:
                df_group.loc[i, 'item'] = item[j]
                continue

    df = df.merge(df_group[['group', 'item']], on='group', how='left')
    
    # 分配结果修改首位排序
    df['pos'] = df.apply(lambda x: x['item_list'].index(x['item']), axis=1)
    
    order = list(range(len(item_dist)))
    df['score_order'] = df.apply(lambda x: '-'.join([x['score_list'][j] for j in [x['pos']] + [i for i in order if i != x['pos']]]), axis=1)
    df['item_order'] = df.apply(lambda x: '-'.join([x['item_list'][j] for j in [x['pos']] + [i for i in order if i != x['pos']]]), axis=1)
    
    # 合并全部结果
    df_none['score_order'] = df_none['score_list'].apply(lambda x: '-'.join(x))
    df_none['item_order'] = df_none['item_list'].apply(lambda x: '-'.join(x))
    
    df_res = pd.concat([df[['uid', 'item_order', 'score_order']], 
                        df_none[['uid', 'item_order', 'score_order']]], axis=0)
    df_res.reset_index(drop=True, inplace=True)
    df_res.rename(columns={'item_order': cols_rep['item_order'], 
                           'score_order': cols_rep['score_order']}, 
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
    
    # group_simp
    cols_score_group = ['score_03', 'score_04']
    coef_score_rule = 1e-10
    qcut_num = [100, 100]
    lp_file = 'flow_dist/data/flow.lp'
    
    ## data read
    df_rule_score = utils.load_data('flow_dist/data/ugrs.txt')
    
    ## LP
    df_res = LP(df_rule_score, 
                cols_score, cols_score_dist, item, ratio_item, 
                coef_flow_shrink, 
                cols_rep, 
                cols_score_group, 
                qcut_num, 
                lp_file, 
                is_rule=is_rule, cols_rule=cols_rule, cols_rule_dist=cols_rule_dist, coef_score_rule=coef_score_rule)
    
    ## data save
    utils.save_pickle(df_res, res_file)
    