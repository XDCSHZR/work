# -*- coding: utf-8 -*-
import pandas as pd
import pulp
import math
import numpy as np

import utils


def LP(df, cols_score,
       cols_score_group,
       list_item, list_ratio_item, coef_flow_shrink,
       lp_file,
       list_code_item,
       dict_cols_rep,
       qcut_num=10, is_rule=0, cols_rule=[], coef_score_rule=1e-10):
    # s.t. 1：规则限定，分数降权
    cols_score_rule = []
    cols_score_group_rule = []
    if is_rule == 1:
        if len(cols_rule) != len(cols_score):
            raise Exception('rules wrong!!!')

        cols_score_rule = [x + '_rule' for x in cols_score]
        cols_score_group_rule = [x + '_rule' for x in cols_score_group]

        for i in range(len(cols_rule)):
            df[cols_score_rule[i]] = df[cols_score[i]] * (1-(1-df[cols_rule[i]])*(1-coef_score_rule))
    else:
        cols_score_rule = cols_score
        cols_score_group_rule = cols_score_group

    # 排序分桶
    for col in cols_score_group_rule:
        utils.quant_cut(df, col, qcut_num)

    cols_score_group_rule_quant = [x + '_quant' for x in cols_score_group_rule]

    qcut_num_carry_int = 1
    qcut_num_copy = qcut_num
    while qcut_num_copy > 1:
        qcut_num_copy /= 10
        qcut_num_carry_int *= 10

    df['group'] = qcut_num_carry_int ** len(cols_score_group_rule_quant)
    for i, col in enumerate(cols_score_group_rule_quant[::-1]):
        df['group'] += df[col].astype(np.int32) * qcut_num_carry_int ** i

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
            df_group['count'].sum() * list_ratio_item[j]) * coef_flow_shrink, 'flow {}'.format(list_item[j])

    # 保存lp文件
    func.writeLP(lp_file)

    # 求解
    func.solve()
    print(pulp.LpStatus[func.status])

    df_group['item'] = '空白组'
    for i in range(rows):
        sum_row = 0
        for j in range(cols):
            sum_row += pulp.value(var_choices[i][j])
            if sum_row > 1:
                raise Exception('flow distribute wrong!!!')
            if pulp.value(var_choices[i][j]) == 1:
                df_group.loc[i, 'item'] = list_item[j]
        if sum_row == 0:
            raise Exception('flow distribute wrong!!!')

    df = df.merge(df_group[['group', 'item']], on='group', how='left')

    # 分配结果按打分排序&编码
    list_df_tmp = []
    for x, y in zip(cols_score, list_item):
        index_item = df[df['item']==y].index.values
        df_tmp = df.loc[index_item, ['uid', x]]
        df_tmp.sort_values(by=x, ascending=False, inplace=True)
        df_tmp.reset_index(drop=True, inplace=True)
        df_tmp.reset_index(inplace=True)
        df_tmp.rename(columns={'index': 'item_score_rank'}, inplace=True)
        df_tmp['item_score_rank'] += 1
        list_df_tmp.append(df_tmp)
    df_tmp_all = pd.concat(list_df_tmp, axis=0)
    df_tmp_all.reset_index(drop=True, inplace=True)
    df = df.merge(df_tmp_all[['uid', 'item_score_rank']], on='uid', how='left')
    df['item_score_rank'].fillna(-1, inplace=True)

    dict_item = {x: y for x, y in zip(list_item, list_code_item)}
    df['item_code'] = df['item'].apply(lambda x: dict_item[x])

    df_res = df[['uid', 'group_index', 'item_score_rank', 'item', 'item_code']]
    df_res.rename(columns={'item_score_rank': dict_cols_rep['item_score_rank'],
                           'item': dict_cols_rep['item'],
                           'item_code': dict_cols_rep['item_code']
                           },
                  inplace=True)

    return df_res
