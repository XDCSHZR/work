#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:01:36 2020

@author: hzr
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
from statsmodels.stats import weightstats
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score


def save_pickle(data, file_path_name):
    with open(file_path_name, 'wb') as f:
        pickle.dump(data, f, protocol=4)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data_pickle = pickle.load(f)
    return data_pickle


def df_des(df):
    df_miss = df.isnull().mean() * 100
    df_miss = pd.DataFrame({'Miss Percent(%)': df_miss})

    return pd.concat([df.describe().T, df_miss], axis=1)


def binning_st(data, feature_name, bins_thresholds):
    '''
    分箱 -- 给定阈值
    
    Parameters
    ----------
    data: pandas.DataFrame
        数据
        
    feature_name: str
        特征名称
    
    bins_thresholds: List
        分箱阈值
    
    Returns
    -------
    data_bins_index: pandas.DataFrame
        分箱结果, 索引标识
    
    bins_thresholds: List
        分箱范围
    '''
    
    data_new = data[[feature_name]].copy()
    
    data_bins_index = pd.DataFrame(pd.cut(data[feature_name], bins_thresholds, labels=False))
    data_new[feature_name+'_bins_index'] = data_bins_index[feature_name]
    
    bins_thresholds = []
    for i in sorted(list(data_bins_index[feature_name].unique())):
        min_i = data_new.loc[data_new[feature_name+'_bins_index']==i, feature_name].min()
        max_i = data_new.loc[data_new[feature_name+'_bins_index']==i, feature_name].max()
        if i == 0:
            bins_thresholds.append(min_i)
            bins_thresholds.append(max_i)
        else:
            bins_thresholds.append(max_i)
    
    return  data_bins_index, bins_thresholds


def binning_ef(data, feature_name, bins_num=5, weight_feature_name=None):
    '''
    分箱 -- 等频
    
    Parameters
    ----------    
    data: pandas.DataFrame
        数据
        
    feature_name: str
        特征名称
    
    bins_num: int
        分箱个数
    
    weight_feature_name: str
        权重变量名称
    
    Returns
    -------
    data_bins_index: pandas.DataFrame
        分箱结果, 索引标识
    
    bins_thresholds: List
        分箱范围
    '''
    
    if weight_feature_name != None:
        data_new = data[[feature_name, weight_feature_name]].copy()
        
        data_new_temp = data_new.sort_values(by=feature_name)
        data_new_temp.reset_index(drop=True, inplace=True)
        data_new_temp[weight_feature_name+'_cumsum'] = data_new_temp[weight_feature_name].cumsum()
        
        bins_thresholds_input = []
        for pcut in [100/bins_num*i for i in range(1, bins_num)]:
            weight_cut = np.nanpercentile(data_new_temp[weight_feature_name+'_cumsum'], pcut)
            fea_cut = data_new_temp[data_new_temp[weight_feature_name+'_cumsum']<=weight_cut].tail(1)[feature_name].values[0]
            bins_thresholds_input.append(fea_cut)
        bins_thresholds_input = [-np.inf] + sorted(list(set(bins_thresholds_input))) + [np.inf]
        
        return binning_st(data_new, feature_name, bins_thresholds_input)
    
    data_new = data[[feature_name]].copy()
    
    data_bins_index = pd.DataFrame(pd.qcut(data_new[feature_name], bins_num, duplicates='drop', labels=False))
    bins_interval = pd.qcut(data_new[feature_name], bins_num, duplicates='drop').value_counts().index
    
    bins_list_order = sorted(bins_interval)
    bins_thresholds = [data_new[feature_name].min()]
    for t in range(0, len(bins_list_order)):
        bins_thresholds.append(bins_list_order[t].right)
            
    return data_bins_index, bins_thresholds


def binning_ed(data, feature_name, bins_num=5):
    '''
    分箱 -- 等距
    
    Parameters
    ----------    
    data: pandas.DataFrame
        数据
        
    feature_name: str
        特征名称
    
    bins_num: int
        分箱个数
    
    Returns
    -------
    data_bins_index: pandas.DataFrame
        分箱结果, 索引标识
    
    bins_thresholds: List
        分箱范围
    '''
    
    data_new = data[[feature_name]].copy()
    
    data_bins_index = pd.DataFrame(pd.cut(data_new[feature_name], bins_num, labels=False))
    bins_interval = pd.cut(data_new[feature_name], bins_num).value_counts().index
    
    bins_list_order = sorted(bins_interval)
    bins_thresholds = [data_new[feature_name].min()]
    for t in range(0, len(bins_list_order)):
        bins_thresholds.append(bins_list_order[t].right)
            
    return data_bins_index, bins_thresholds


def binning_tree(data, feature_name, label_name, 
                 max_leafNode_num=5, min_leafSample_percent=0.05, 
                 weight_feature_name=None):
    '''
    分箱 -- CART
    
    Parameters
    ----------
    data: pandas.DataFrame
        数据
        
    feature_name: str
        特征名称
    
    label_name: str
        标签名称
    
    max_leafNode_num: int
        最大叶节点个数(分箱数)
    
    min_leafSample_percent: float
        最小叶节点样本数百分比
        
    weight_feature_name: str
        权重变量名称
    
    Returns
    -------
    data_bins_index: pandas.DataFrame
        分箱结果, 索引标识
    
    bins_thresholds: List
        分箱范围
    '''
    
    def get_median(data, feature_name):
        '''
        相邻特征值中位数, 用于切分数据
    
        Parameters
        ----------    
        data: pandas.DataFrame
            数据
        
        feature_name: str
            特征名称
    
        Returns
        -------
        list_vals_median: List
            相邻特征值中位数列表
        '''
        
        list_vals = sorted(data[feature_name].unique())
        
        list_vals_median = []
        for i in range(len(list_vals)-1):
            vals_median = (list_vals[i]+list_vals[i+1]) / 2
            list_vals_median.append(vals_median)
        
        return list_vals_median
    
    def get_Gini(data, feature_name, label_name):
        '''
        当前数据集在该特征下的基尼系数
    
        Parameters
        ----------    
        data: pandas.DataFrame
            数据
        
        feature_name: str
            特征名称
        
        label_name: str
            标签名称
        
        Returns
        -------
        Gini: float
            基尼系数
        '''
        
        list_label_val = data[label_name].unique()
        D = data['weight'].sum()
        Gini_D = 1
        
        for k in list_label_val:
            C_k = data[data[label_name]==k]['weight'].sum()
            Gini_D -= (C_k/D) ** 2
            
        return Gini_D
    
    def CART_once(data, feature_name, label_name, 
                  min_leafSample):
        '''
        CART方法选择特征最好切分点, 单次切分
    
        Parameters
        ----------    
        data: pandas.DataFrame
            数据
        
        feature_name: str
            特征名称
        
        label_name: str
            标签名称
    
        min_leafSample: int/float
            最小叶节点样本数
        
        Returns
        -------
        A_best: float
            最优切分点
        '''
        
        list_vals_median = get_median(data, feature_name)
        val_min = data[feature_name].min()
        val_max = data[feature_name].max()
        D = data['weight'].sum()
        D_min = min_leafSample
        Gini_D = get_Gini(data, feature_name, label_name)
        
        Gain_D_A_best = 0
        A_best = -9999
        
        for A in list_vals_median:
            data_small = data[data[feature_name]<A]
            data_large = data[data[feature_name]>A]
            
            D1 = data_small['weight'].sum()
            D2 = data_large['weight'].sum()
            
            if D1 < D_min or D2 < D_min:
                continue
            
            Gini_D1 = get_Gini(data_small, feature_name, label_name)
            Gini_D2 = get_Gini(data_large, feature_name, label_name)
            
            Gini_D_A = D1 / D * Gini_D1 + D2 / D * Gini_D2
            
            Gain_D_A = Gini_D - Gini_D_A
            
            if Gain_D_A > Gain_D_A_best:
                Gain_D_A_best = Gain_D_A
                A_best = A
                
        return val_min, val_max, A_best, Gini_D, Gain_D_A_best
   
    def CART_single_feature_loop(data, feature_name, label_name, 
                                 min_leafSample, list_A):
        '''
        CART方法选择特征切分点, 循环
    
        Parameters
        ----------    
        data: pandas.DataFrame
            数据
        
        feature_name: str
            特征名称
        
        label_name: str
            标签名称
    
        min_leafSample: int\float
            最小叶节点样本数
        
        list_A: List[int\float]
            特征划分点值列表
        
        Returns
        -------
        None
        '''
        
        val_min, val_max, A_best, Gini_D, Gain_D_A_best = CART_once(data, feature_name, label_name, min_leafSample)
        
        if A_best == -9999:
            return
        else:
            list_A.append(A_best)
        
        data_small = data[data[feature_name]<A_best]
        data_large = data[data[feature_name]>A_best]
        
        if data_small.shape[0] >= min_leafSample * 2:
            CART_single_feature_loop(data_small, feature_name, label_name, min_leafSample, list_A)
        
        if data_large.shape[0] >= min_leafSample * 2:
            CART_single_feature_loop(data_large, feature_name, label_name, min_leafSample, list_A)
    
    def CART_binning_postCombine_simple(data, feature_name, label_name, 
                                        list_split_val, max_leafNode_num=5):
        '''
        CART分箱"后合并"方法, 合并相邻区间, 选择基尼系数增益最大的, 循环直至满足最大叶节点个数
    
        Parameters
        ----------
        data: pandas.DataFrame
            数据
        
        feature_name: str
            特征名称
        
        label_name: str
            标签名称
            
        list_split_val: List[int\float]
            特征划分点值列表
        
        max_leafNode_num: int
            最大叶节点个数
        
        Returns
        -------
        list_split_val_final: List[int\float]
            特征划分点值列表(合并后)
        '''
        
        list_split_val = [-np.inf] + list_split_val + [np.inf]
        leafNode_num = len(list_split_val) - 1
        list_split_val_final = list_split_val.copy()
        
        list_leafNode_sample = []
        list_Gini_leafNode = []
        for i in range(leafNode_num):
            data_leafNode = data[(data[feature_name]>list_split_val[i])&
                                 (data[feature_name]<list_split_val[i+1])]
            list_leafNode_sample.append(data_leafNode['weight'].sum())
            list_Gini_leafNode.append(get_Gini(data_leafNode, feature_name, label_name))
        
        list_Gini_D_A = []
        list_Gini_D = []
        list_Gain_D_A = []
        for i in range(leafNode_num-1):
            data_D = data[(data[feature_name]>list_split_val[i])&
                          (data[feature_name]<list_split_val[i+2])]
            list_Gini_D.append(get_Gini(data_D, feature_name, label_name))
            
            D1 = list_leafNode_sample[i]
            D2 = list_leafNode_sample[i+1]
            D = D1 + D2
            Gini_D1 = list_Gini_leafNode[i]
            Gini_D2 = list_Gini_leafNode[i+1]
            list_Gini_D_A.append(D1/D*Gini_D1+D2/D*Gini_D2)
            
            list_Gain_D_A.append(list_Gini_D_A[i]-list_Gini_D[i])
        
        while leafNode_num > max_leafNode_num and leafNode_num > 3:
            pos_del = np.argmax(list_Gain_D_A)
            list_split_val_final.pop(pos_del+1)
            
            list_leafNode_sample[pos_del] += list_leafNode_sample[pos_del+1]
            list_leafNode_sample.pop(pos_del+1)
            
            list_Gini_leafNode[pos_del] = list_Gini_D[pos_del]
            list_Gini_leafNode.pop(pos_del+1)
            
            if pos_del == len(list_Gain_D_A) - 1:
                data_D = data[(data[feature_name]>list_split_val_final[pos_del-1])&
                              (data[feature_name]<list_split_val_final[pos_del+1])]
                list_Gini_D[pos_del-1] = get_Gini(data_D, feature_name, label_name)
                list_Gini_D.pop()
                
                D1 = list_leafNode_sample[pos_del-1]
                D2 = list_leafNode_sample[pos_del]
                D = D1 + D2
                Gini_D1 = list_Gini_leafNode[pos_del-1]
                Gini_D2 = list_Gini_leafNode[pos_del]
                list_Gini_D_A[pos_del-1] = D1 / D * Gini_D1 + D2 / D * Gini_D2
                list_Gini_D_A.pop()
            
                list_Gain_D_A[pos_del-1] = list_Gini_D_A[pos_del-1] - list_Gini_D[pos_del-1]
                list_Gain_D_A.pop()
            elif pos_del == 0:
                data_D = data[(data[feature_name]>list_split_val_final[pos_del])&
                              (data[feature_name]<list_split_val_final[pos_del+2])]
                list_Gini_D[pos_del] = get_Gini(data_D, feature_name, label_name)
                list_Gini_D.pop(pos_del+1)
                
                D1 = list_leafNode_sample[pos_del]
                D2 = list_leafNode_sample[pos_del+1]
                D = D1 + D2
                Gini_D1 = list_Gini_leafNode[pos_del]
                Gini_D2 = list_Gini_leafNode[pos_del+1]
                list_Gini_D_A[pos_del] = D1 / D * Gini_D1 + D2 / D * Gini_D2
                list_Gini_D_A.pop(pos_del+1)
            
                list_Gain_D_A[pos_del] = list_Gini_D_A[pos_del] - list_Gini_D[pos_del]
                list_Gain_D_A.pop(pos_del+1)
            else:
                data_D_1 = data[(data[feature_name]>list_split_val_final[pos_del-1])&
                                (data[feature_name]<list_split_val_final[pos_del+1])]
                data_D_2 = data[(data[feature_name]>list_split_val_final[pos_del])&
                                (data[feature_name]<list_split_val_final[pos_del+2])]
                list_Gini_D[pos_del-1] = get_Gini(data_D_1, feature_name, label_name)
                list_Gini_D[pos_del] = get_Gini(data_D_2, feature_name, label_name)
                list_Gini_D.pop(pos_del+1)
                
                D1_1 = list_leafNode_sample[pos_del-1]
                D2_1 = list_leafNode_sample[pos_del]
                D1_2 = list_leafNode_sample[pos_del]
                D2_2 = list_leafNode_sample[pos_del+1]
                D_1 = D1_1 + D2_1
                D_2 = D1_2 + D2_2
                Gini_D1_1 = list_Gini_leafNode[pos_del-1]
                Gini_D2_1 = list_Gini_leafNode[pos_del]
                Gini_D1_2 = list_Gini_leafNode[pos_del]
                Gini_D2_2 = list_Gini_leafNode[pos_del+1]
                list_Gini_D_A[pos_del-1] = D1_1 / D_1 * Gini_D1_1 + D2_1 / D_1 * Gini_D2_1
                list_Gini_D_A[pos_del] = D1_2 / D_2 * Gini_D1_2 + D2_2 / D_2 * Gini_D2_2
                list_Gini_D_A.pop(pos_del+1)
            
                list_Gain_D_A[pos_del-1] = list_Gini_D_A[pos_del-1] - list_Gini_D[pos_del-1]
                list_Gain_D_A[pos_del] = list_Gini_D_A[pos_del] - list_Gini_D[pos_del]
                list_Gain_D_A.pop(pos_del+1)
                
            leafNode_num -= 1
            
        return list_split_val_final
    
    list_valid_cols = [feature_name, label_name]
    if weight_feature_name != None:
        list_valid_cols.append(weight_feature_name)
        data_new = data[list_valid_cols].copy()
        data_new.rename(columns={weight_feature_name: 'weight'}, inplace=True)
    else:
        data_new = data[list_valid_cols].copy()
        data_new['weight'] = 1
        
    min_leafSample = data_new['weight'].sum() * min_leafSample_percent
    list_split_val = []
    
    CART_single_feature_loop(data_new, feature_name, label_name, 
                             min_leafSample, list_split_val)
    list_split_val.sort()
    
    list_split_val_final = CART_binning_postCombine_simple(data_new, feature_name, label_name, 
                                                           list_split_val, 
                                                           max_leafNode_num=max_leafNode_num)
    
    data_bins_index, bins_thresholds = binning_st(data, feature_name, list_split_val_final)
    
    return data_bins_index, bins_thresholds


def binning(data, feature_name, 
            miss_val=-1,  weight_feature_name=None, label_name=None, 
            method='ef', parameter=None, dec=2):
    '''
    分箱
    
    Parameters
    ----------
    data: pandas.DataFrame
        数据
        
    feature_name: str
        特征名称
    
    miss_val: float/int
        特征缺失填充值
    
    weight_feature_name: str
        权重特征名称, 有监督分箱方法(tree...)必要参数
    
    label_name: str
        标签名称, 有监督分箱方法(tree...)必要参数
    
    method: str
        分箱方法: 
            'st'   -- 给定阈值
            'ef'   -- 等频
            'ed'   -- 等距
            'tree' -- CART分箱
    
    parameter: dict
        参数, 具体对应不同分箱方法:
            'st'   -- {'bins_thresholds': xxx} 
            'ef'   -- {'bins_num': xxx}
            'ed'   -- {'bins_num': xxx}
            'tree' -- {'max_leafNode_num': xxx, 'min_leafSample_percent': xxx}
        
    dec: int 
        分箱范围精确位数
    
    Returns
    -------
    df_bins_index: pandas.DataFrame
        分箱结果, 原数据和分箱索引
    
    list_bins: List
        分箱范围
    '''
    
    list_bins = []
    list_valid_cols = [x for x in [feature_name, weight_feature_name, label_name] if x != None]
    df_bins_index = data[list_valid_cols].copy()
    
    # miss
    df_bins_index[feature_name+'_bins_index'] = miss_val
    index_unmiss = df_bins_index[(df_bins_index[feature_name].notna())&
                                 (df_bins_index[feature_name]!=miss_val)].index
    
    # method
    if method == 'st':
        df_bins_index_unmiss, list_bins_thresholds_unmiss = \
            binning_st(df_bins_index.loc[index_unmiss, :], feature_name, **parameter)
    elif method == 'ef':
        parameter.update({'weight_feature_name': weight_feature_name})
        df_bins_index_unmiss, list_bins_thresholds_unmiss = \
            binning_ef(df_bins_index.loc[index_unmiss, :], feature_name, **parameter)
    elif method == 'ed':
        df_bins_index_unmiss, list_bins_thresholds_unmiss = \
            binning_ed(df_bins_index.loc[index_unmiss, :], feature_name, **parameter)
    elif method == 'tree':
        parameter.update({'weight_feature_name': weight_feature_name})
        df_bins_index_unmiss, list_bins_thresholds_unmiss = \
            binning_tree(df_bins_index.loc[index_unmiss, :], feature_name, label_name, **parameter)
    
    df_bins_index.loc[df_bins_index_unmiss.index, feature_name+'_bins_index'] = df_bins_index_unmiss[feature_name]
    
    list_bins_sort = sorted(df_bins_index[feature_name+'_bins_index'].unique())
    for index in list_bins_sort:
        if index == miss_val:
            list_bins.append('['+str(miss_val)+', '+str(miss_val)+']')
        elif index == 0:
            list_bins.append('['+str(round(list_bins_thresholds_unmiss[int(index)], dec))+', '+str(round(list_bins_thresholds_unmiss[int(index)+1], dec))+']')
        else:
            list_bins.append('('+str(round(list_bins_thresholds_unmiss[int(index)], dec))+', '+str(round(list_bins_thresholds_unmiss[int(index)+1], dec))+']')
    
    return df_bins_index, list_bins


def plotfeats(data, feats, kind, label=None, label_pos=None, cols=4, path=None, pic_name=None):
    '''
    批量绘图函数
    
    Parameters
    ----------
    data: pandas.DataFrame
        绘图数据
        
    feats: List[str]
        特征名称列表
            
    kind: str 
        绘图格式: 
            'pair_Xs_num' - 多变量图
            'hist_X_num' - 概率密度直方图
            'scatter_X_y_num' - 散点图 
            'box_X_num' - 箱线图（连续特征） 
            'box_X_y_cat' - 箱线图（离散特征）
            'countplot' - 数量直方图
    
    label: str
        标签: 默认为'None', kind str中包含y的为必须参数
    
    label_pos: str/int
        正样本标签: 默认为'None', 主要用于二分类, kind str中包含y时使用
    
    cols: int 
        每行绘制的图数量
    
    path: str
        图片存储路径
    
    pic_name: str
        图片名称
    
    Returns
    -------
    None
    '''
    
    plt.rcParams['font.sans-serif'] = ['simhei']
    plt.rcParams['font.serif'] = ['simhei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(font_scale=1.5)
    sns.set_style('darkgrid', {'font.sans-serif':['simhei', 'Droid Sans Fallback']})
    
    rows = int(np.ceil((len(feats))/cols))
    if rows == 1 and len(feats) < cols:
        cols = len(feats)
    if rows == 1 and cols == 1:
        fig = plt.figure(figsize=(12, 8), dpi=60)
    else:
        fig = plt.figure(figsize=(rows*5, cols*5), dpi=60)
    
    if kind == 'pair_Xs_num':
        if isinstance(feats, list) and len(feats) > 1:
            sns.pairplot(data[feats], vars=feats)
        else:
            raise Exception('Invalid Parameter!')
    else:
        for i, f in enumerate(feats):
            axes = fig.add_subplot(rows, cols, i+1)
            
            data_f = data[data[f].notna()]
            
            if kind == 'hist_X_num':
                (mu, sigma) = norm.fit(data_f[f])
                sns.distplot(data_f[f], fit=norm, ax=axes)
                plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f})'.format(mu, sigma)], loc='best')
                plt.xlabel(f)
                plt.ylabel('Frequency')
            elif kind == 'scatter_X_y_num' and label != None:
                sns.scatterplot(x=f, y=label, data=data_f, ax=axes)
            elif kind == 'box_X_num':
                sns.boxplot(y=data_f[f], ax=axes)
            elif kind == 'box_X_y_cat' and label != None:
                sns.boxplot(x=f, y=label, data=data_f, ax=axes)
            elif kind == 'countplot':
                if label_pos is not None:
                    df_ct = pd.crosstab(data[f], data[label])
                    ax = sns.countplot(hue=label, x=f, data=data, order=df_ct.index, ax=axes)
                    for p, count_pos in zip(ax.patches, df_ct[label_pos].values):
                        ax.annotate('pos:{cp}'.format(cp=count_pos), 
                                    (p.get_x()+0.15, p.get_height()))
                else:
                    sns.countplot(hue=label, x=f, data=data, ax=axes)
            else:
                raise Exception('Invalid Parameter!')
    
    if path is not None and pic_name is not None:
        plt.savefig(path+pic_name+'.jpg')
    
    plt.show()


def one_hot_encoder(data, feat):
    '''
    独热编码
    
    Parameters
    ----------    
    data: pandas.DataFrame
        数据
    
    feat: str
        特征名称
    
    Returns
    -------
    df_ohe_feat: pandas.DataFrame
        独热编码结果
    
    '''
    
    df = data[[feat]].copy()
    df[feat] = df[feat].astype(str)
    
    enc = OneHotEncoder(categories='auto')
    clf_ohe = enc.fit(df)
    
    ohe_feat_data = clf_ohe.transform(df).toarray()
    ohe_feat_col = [feat+'_'+str(x) for x in list(clf_ohe.categories_[0])]
    
    df_ohe_feat = pd.DataFrame(ohe_feat_data)
    df_ohe_feat.columns = ohe_feat_col
    
    return df_ohe_feat


def hypothesis_testing(data, feats, label, method='f_classif', num=1, P_threshold=0.05):
    '''
    使用假设检验进行特征选择
    
    Parameters
    ----------
    data: pandas.DataFrame
        数据
        
    feats: List
        特征名称列表
    
    label: str
        标签名称
    
    method: str
        假设检验方法:
            'f_classif' - F检验（分类）
            'chi2' - 卡方检验（分类）
            'f_regression' - F检验（回归）
    
    num: int
        选择的特征个数
    
    P_threshold: float
        P值阈值, 显著性水平
    
    Returns
    -------
    df: pandas.DataFrame
        各特征对应的分数、P值以及是否被选中
    '''
    
    df = pd.DataFrame(columns=['feature', 'score', 'P_value', 'is_selected_score', 'is_selected_PValue'])
    
    if method == 'chi2':
        func = chi2
    elif method == 'f_classif':
        func = f_classif
    elif method == 'f_regression':
        func = f_regression
    else:
        raise Exception('Invalid Parameter!')
    selector = SelectKBest(func, k=num).fit(data[feats], data[label])
    feats_score = selector.scores_
    feats_PValue = selector.pvalues_
    feats_isSelected_index = selector.get_support(True)
    
    df['feature'] = feats
    df['score'] = feats_score
    df['P_value'] = feats_PValue
    df['is_selected_score'] = 0
    df['is_selected_PValue'] = 0
    
    df.loc[feats_isSelected_index, 'is_selected_score'] = 1
    df.loc[df['P_value']<=P_threshold, 'is_selected_PValue'] = 1
    df.sort_values(by=['score', 'P_value'], ascending=[False, False], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def cal_woe_iv(data, feature_bins_index_name, label_name, bins_range, 
               is_category=0, label_true_val=1, label_false_val=0, 
               weight_feature_name=None):
    '''
    二分类woe, iv计算
    
    Parameters
    ----------
    data: pandas.DataFrame
        数据
        
    feature_bins_index_name: str
        特征名称 -- 特征分箱后的索引
    
    label_name: str
        标签名称
    
    bins_range: List[str]
        分箱范围列表
    
    is_category: int
        是否为定性特征
    
    label_true_val: str\int\float
        正样本标签值
    
    label_false_val: str\int\float
        负样本标签值
    
    weight_feature_name: str
        权重变量名称
        
    Returns
    -------
    df_res: pandas.DataFrame
        该特征的各特征值对应正负标签的count, rate, woe, iv
    '''
    
    list_valid_cols = [feature_bins_index_name, label_name]
    if weight_feature_name != None:
        list_valid_cols.append(weight_feature_name)
        df = data[list_valid_cols].copy()
        df.rename(columns={weight_feature_name: 'weight'}, inplace=True)
    else:
        df = data[list_valid_cols].copy()
        df['weight'] = 1
    
    if isinstance(label_true_val, (int, float)):
        df[label_name] = df[label_name].astype(float).astype(str)
        label_true_str = str(float(label_true_val))
        label_false_str = str(float(label_false_val))
    else:
        label_true_str = label_true_val
        label_false_str = label_false_val
    
    # 交叉表统计各特征值在"正负"标签下的数量
    df_crossCount = pd.crosstab(df[feature_bins_index_name], df[label_name], 
                                values=df['weight'], aggfunc=np.sum)
    df_crossCount.reset_index(inplace=True)
    
    df_na = df[df[feature_bins_index_name].isna()]
    if df_na.shape[0] > 0:    
        df_nan = pd.DataFrame({feature_bins_index_name: [np.nan], 
                               label_false_str: [df_na[df_na['label']==label_false_str]['label'].count()], 
                               label_true_str: [df_na[df_na['label']==label_true_str]['label'].count()]})
        df_crossCount = pd.concat([df_nan, df_crossCount], axis=0)
        df_crossCount.reset_index(drop=True, inplace=True)
    
    df_crossCount['total'] = df_crossCount.iloc[:, df_crossCount.columns!=feature_bins_index_name].sum(axis=1)
    df_sum = pd.DataFrame(df_crossCount.sum()).T
    df_sum.loc[0, feature_bins_index_name] = '合计'
    
    # “正”，“负”占比
    yes_i = df_crossCount.loc[:, label_true_str]
    df_crossCount['pos_rate'] = yes_i / df_sum.loc[0, label_true_str]
        
    no_i = df_crossCount.loc[:, label_false_str]
    df_crossCount['neg_rate'] = no_i / df_sum.loc[0, label_false_str]
        
    # woe
    df_crossCount['woe'] = np.log(df_crossCount['pos_rate']/df_crossCount['neg_rate'])
    df_crossCount = df_crossCount.replace({'woe': {np.inf: 0, -np.inf: 0}})
        
    # iv
    df_crossCount['iv'] = (df_crossCount['pos_rate']-df_crossCount['neg_rate']) * df_crossCount['woe']
    df_crossCount['iv'] = abs(df_crossCount['iv'])
    
    df_res = pd.DataFrame()
    df_res['bin_index'] = df_crossCount[feature_bins_index_name]
    df_res['bin_range'] = bins_range
    df_res['good'] = df_crossCount[label_true_str]
    df_res['bad'] = df_crossCount[label_false_str]
    df_res['total'] = df_crossCount['total']
    df_res['bad_rate(%)'] = round(df_res['bad']/df_res['total']*100, 1)
    df_res['woe'] = df_crossCount['woe']
    df_res['iv'] = df_crossCount['iv']
    df_res['lift'] = round(df_res['bad']/df_res['total']/((df_sum[label_false_str]/df_sum['total'])[0]), 2) 
    
    df_res_sum = pd.DataFrame()
    df_res_sum['bin_index'] = df_sum[feature_bins_index_name]
    if is_category == 1:
        df_res_sum['bin_range'] = [','.join(bins_range)]
    else:
        df_res_sum['bin_range'] = [bins_range[0].split(',')[0]+','+bins_range[-1].split(',')[1]]
    df_res_sum['good'] = df_sum[label_true_str].astype(float)
    df_res_sum['bad'] = df_sum[label_false_str].astype(float)
    df_res_sum['total'] = df_sum['total'].astype(float)
    df_res_sum['bad_rate(%)'] = round(df_res_sum['bad']/df_res_sum['total']*100, 1)
    df_res_sum['woe'] = [-9999]
    df_res_sum['iv'] = [df_crossCount['iv'].sum()]
    df_res_sum['lift'] = [-9999]
    
    df_res = pd.concat([df_res, df_res_sum], axis=0)
    df_res.reset_index(drop=True, inplace=True)
    
    return df_res


def check_corr(data, variables, weight=None, miss=True, method='pearson', 
               size_pic=20, size_font=20, path=None, pic_name=None):
    '''
    相关性分析
    (用于特征筛选时可结合缺失率, 树形模型重要度, 与标签的相关性等, 剔除相关性较高的两个特征中的一个)
    
    Parameters
    ----------
    data: pandas.DataFrame
        数据
    
    variables: List[str]
        特征名称列表
    
    weight: str
        权重特征名称
    
    miss: bool
        数据是否含有缺失值, 仅对加权数据起作用
    
    method: str or function()
        度量方法: 
            'pearson'  -- 皮尔逊相关系数(线性) 
            'spearman' -- 斯皮尔曼相关系数(线性, 非线性) 
            'kendall'  -- 肯德尔相关系数(非参数, 线性, 非线性)
            or callable(), 仅对非权重数据有效
    
    size_pic: int
        相关性图片大小
        
    size_font: int
        相关性图片中数字大小
        
    path: str
        图片存储路径
    
    pic_name: str
        图片存储名称
    
    Returns
    -------
    corr: pandas.DataFrame
        相关系数
    '''    
    
    def pearson_corr_x_y(data, variables, index_x, index_y, weight):
        '''
        皮尔逊相关系数
        
        Parameters
        ----------
        data: pandas.DataFrame
            数据
        
        variables: List[str]
            特征名称列表
        
        index_x: int
            x特征名称索引
        
        index_y: int
            y特征名称索引
        
        weight: str
            权重变量名称
        
        Returns
        -------
        corr: pandas.DataFrame
            相关系数
        '''
        stats_summary_temp = weightstats.DescrStatsW(data[[variables[index_x], variables[index_y]]], weights=data[weight])
        
        return stats_summary_temp.corrcoef[0, 1]
    
    def spearman_corr_x_y(data, variables, x, y, weight):
        '''
        斯皮尔曼相关系数
        
        Parameters
        ----------
        data: pandas.DataFrame
            数据
        
        variables: List[str]
            特征名称列表
        
        index_x: int
            x特征名称索引
        
        index_y: int
            y特征名称索引
        
        weight: str
            权重变量名称
        
        Returns
        -------
        corr: pandas.DataFrame
            相关系数
        '''
        
        data[variables[x]+'_rank'] = data[variables[x]].rank()
        data[variables[y]+'_rank'] = data[variables[y]].rank()
        
        d = ((data[variables[x]+'_rank']-data[variables[y]+'_rank'])**2*data[weight]).sum()
        N = data[weight].sum()
        
        return 1 - 6 * d / (N * (N**2 - 1))
        
    def kendall_corr_x_y(data, variables, x, y, weight):
        '''
        肯德尔相关系数
        
        Parameters
        ----------
        data: pandas.DataFrame
            数据
        
        variables: List[str]
            特征名称列表
        
        index_x: int
            x特征名称索引
        
        index_y: int
            y特征名称索引
        
        weight: str
            权重变量名称
        
        Returns
        -------
        corr: pandas.DataFrame
            相关系数
        '''
        
        len_data = data.shape[0]
        P = 0
        Q = 0
        T = 0
        U = 0
        index = list(data.index)
        for i in range(len_data-1):
            for j in range(i+1, len_data):
                x_c = data.loc[index[i], variables[x]] - data.loc[index[j], variables[x]]
                y_c = data.loc[index[i], variables[y]] - data.loc[index[j], variables[y]]
                x_y_c = x_c * y_c
                count = data.loc[index[i], weight] * data.loc[index[j], weight]
                
                if x_y_c > 0:
                    P += count
                elif x_y_c < 0:
                    Q += count
                    
                if x_c == 0 and y_c != 0:
                    T += count
                elif x_c != 0 and y_c == 0:
                    U += count
                    
        return (P-Q) / np.sqrt((P+Q+T)*(P+Q+U))
    
    corr = None
    
    if weight is None:
        corr = data[variables].corr(method=method)
    else:
        df = data[variables+[weight]].copy()
        if method == 'pearson':
            if miss == False:
                stats_summary = weightstats.DescrStatsW(df[variables], weights=df[weight])
                corr = stats_summary.corrcoef
            else:
                func = pearson_corr_x_y
        elif method == 'spearman':
            func = spearman_corr_x_y
        elif method == 'kendall':
            func = kendall_corr_x_y
        else:
            raise Exception('Invalid Parameter!')
        
        if corr == None:
            corr = np.identity(len(variables))
            for i in range(len(variables)-1):
                for j in range(i+1, len(variables)):
                    print(i, j)
                    df_temp = df[[variables[i], variables[j], weight]].copy()
                    df_temp = df_temp[(df_temp[variables[i]].notna())&
                                      (df_temp[variables[j]].notna())]
                    corr[i, j] = corr[j, i] = func(df_temp, variables, i, j, weight)
         
        corr = pd.DataFrame(corr, columns=variables, index=variables)
        
    corr.replace({np.nan: 0}, inplace=True)
    
    xticks = list(corr.index)
    yticks = list(corr.index)
    
    fig = plt.figure(figsize=(size_pic, size_pic), dpi=60)
    axes = fig.add_subplot(1, 1, 1)
    sns.heatmap(corr, annot=True, ax=axes, 
                annot_kws={'size': size_font, 'weight': 'bold', 'color': 'green'})
    axes.set_xticklabels(xticks, rotation=90, size=15)
    axes.set_yticklabels(yticks, rotation=0, size=15)
    
    if path is not None and pic_name is not None:
        plt.savefig(path+pic_name+'.png')
    
    plt.show()
    
    return corr


def cal_xgbClassifier_imp(X_train, y_train, obj='binary:logistic', 
                          best_para={}, weight_train=None):
    '''
    xgboost特征重要度计算
    用于特征筛选时需剔除重要度较低的特征, 后续再跑模型再剔除直至满足要求
    
    Parameters
    ----------
    X_train: pd.DataFrame
        训练数据特征
    
    y_train: pd.DataFrame
        训练数据标签
    
    obj: str
        模型学习目标:
            'binary:logistic': 二分类逻辑回归
            'multi:softmax': 多分类softmax
    
    best_para: dict
        模型最好参数
        
    weight_train: pd.DataFrame
        训练数据权重
    
    Returns
    -------
    ft_imp: pd.DataFrame
        特征重要度
    '''
    
    clf_XGB = XGBClassifier(objective=obj, **best_para)
    
    if weight_train is None:
        clf_XGB.fit(X=X_train.values, y=y_train.values)
    else:
        clf_XGB.fit(X=X_train.values, y=y_train.values, sample_weight=weight_train.values)
    
    ft_imp = pd.DataFrame(clf_XGB.feature_importances_, index=X_train.columns).sort_values(0, ascending=False)
    
    return ft_imp


def auc_score_cv(model, data_X, data_y, cv=5):
    '''
    auc分数交叉验证
    
    Parameters
    ----------
    model: callable object 
        估计器函数
        
    data_X: numpy.ndarray
        数据特征
    
    data_y: numpy.ndarray
        数据标签
    
    labels_class: List
        标签类别列表
    
    cv: int
        交叉验证折数
    
    Returns
    -------
    score_avg: float
        平均auc分数
    '''
    
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=2021)
    score_sum = 0
    k = 0
    
    for tdi, vdi in kfold.split(data_X, data_y):
        k = k + 1
            
        train_X = data_X[tdi, :]
        eval_X = data_X[vdi, :]
        train_y = data_y[tdi, :]
        eval_y = data_y[vdi, :]
        
        model.fit(train_X, train_y)
        clf_cv = model.predict_proba(eval_X)
        score = roc_auc_score(eval_y, clf_cv[:, 1])
            
        print('cv:{cv}, auc score:{s}'.format(cv=k, s=score))
        score_sum = score_sum + score
        
    score_avg = score_sum / 5
    
    return score_avg


def PR_threshold(y_true, y_pred_prob, y_all=None, 
                 threshold_bottom=0.0, threshold_top=1.0, threshold_interval=0.01, dec=2):
    '''
    binary classification precision, recall, count(True) for all threshold
    :param y_true: numpy.ndarray 1D
    :param y_pred_prob: numpy.ndarray 1D
    :param y_all: int
    :param threshold_bottom: float
    :param threshold_top: float
    :param threshold_interval: float
    :return: pandas.DataFrame
    '''
    
    list_t = []
    bottom = threshold_bottom
    while bottom <= threshold_top:
        list_t.append(bottom)
        bottom += threshold_interval
        bottom = round(bottom, dec)
    
    list_df_pr = []
    
    try:
        with tqdm(list_t) as t:
            for i in t:
                test_p = np.array([int(x) for x in y_pred_prob >= i])
                cr = classification_report(y_true, test_p, output_dict=True)
                p = cr['1']['precision']
                r = cr['1']['recall']
                c = test_p.sum()
                r_a = c * p / y_all if y_all != None else r
                f1_a = 2 * p * r_a / (p+r_a)
                df_pr_temp = pd.DataFrame({'threshold': [i], 
                                           'precision': [p], 
                                           'recall': [r], 
                                           'count': [c], 
                                           'recall_all': [r_a], 
                                           'F1_all': [f1_a]})
                list_df_pr.append(df_pr_temp)
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()

    df_pr = pd.concat(list_df_pr, axis=0)
    df_pr.reset_index(drop=True, inplace=True)
    
    return df_pr
