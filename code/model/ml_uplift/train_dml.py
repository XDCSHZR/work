import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import econml
from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML
import numpy as np
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, clone
import lightgbm as lgb
import seaborn as sns
import glob
import os
import joblib
import cloudpickle
from reg_wrapper import RegressionWrapper
import argparse


LABLE_ENCODE_DICT = {
    'xxx':5,
    'xxx':2,
    'xxx':0,
    'xxx':1, 
    'xxx':3, 
    'xxx':4 }
# 先只建模这6款产品

TRAIN_FILE_DATE = ['2023-08-28', '2023-09-04', '2023-09-11', '2023-09-18', '2023-09-25']
TEST_FILE_DATE = ['2023-10-02', '2023-10-09']

# cal pmg score 
def pmg(df_summary):
    """
    df_summary: columns: ['treatment', 'predict_treatment', 'label']
    """
    df_summary['if_same'] = (df_summary['predict_treatment'] == df_summary['treatment']).astype(int)
    df_overlap_summary = df_summary[df_summary['if_same'] == 1].groupby('predict_treatment') \
        .agg({'label':'mean'}) \
        .rename(columns={'label':'mean'}) \
        .reset_index()
    df_stg_summary = df_summary.groupby('predict_treatment')\
        .count().reset_index() \
        .rename(columns={'treatment':'num_stg_treatment'})[['predict_treatment','num_stg_treatment']]
    df_finnal_summary = df_overlap_summary.merge(df_stg_summary, how = 'inner', on = 'predict_treatment')
    total_gain = sum(df_finnal_summary['mean'] * df_finnal_summary['num_stg_treatment'])
    total_count = df_finnal_summary.num_stg_treatment.sum()
    avg_gain = total_gain/total_count
    base = df_summary.label.mean()
    gain = (avg_gain - base)/base

    return gain

# dml model output result postprocess
def process_dml_result(df_test, predict_result):
    """
    df_test: DataFrame of test data
    predict_result: dml model output
    """
    df_test_label = df_test[['t1.product', 't1.label_purchase']].rename(
        columns={'t1.product': 'treatment', 't1.label_purchase': 'label'})
    zeros = np.zeros(predict_result.shape[0])
    res_tmp = np.insert(predict_result, 0, zeros, axis=1)
    res_tmp = pd.DataFrame(res_tmp) 
    res_tmp['predict_treatment'] = res_tmp.apply(
        lambda row: sorted(dict(row).items(), key=lambda x: x[1], reverse=True)[0][0], axis=1)

    df_summary = pd.concat([res_tmp, df_test_label], axis=1)
    return df_summary
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dml')
    parser.add_argument('--cols', type=str, default="_50", help='column file')  # "_manual"
    parser.add_argument('--version', type=str, default='1_3', help='model version')

    args, unknown = parser.parse_known_args()
    
    # load data and preprocess
    print('-------------------------\n loading data... \n-------------------------\n')
    # cols = pd.read_csv('cols_select1.txt', header=None)
    cols = pd.read_csv('cols_select{}.txt'.format(args.cols), header=None)
    cols = list(cols[0])

    train_file_list = ['train_dataset_{}.txt'.format(i) for i in TRAIN_FILE_DATE]
    df = pd.concat([pd.read_csv(i, sep='\t') for i in train_file_list], ignore_index=True)
    df = df[cols]
    df = df.fillna(0)

    label_encode_dict = LABLE_ENCODE_DICT
    df = df[df['t1.product'].isin(list(label_encode_dict.keys()))].reset_index(drop=True)
    df['t1.product'] = df['t1.product'].apply(lambda x: label_encode_dict[x])
    T, Y, X = df['t1.product'], df['t1.label_purchase'], df.iloc[:, :-2]
    print(df.describe())

    
    # build model and train
    print('-------------------------\n training... \n-------------------------\n')
    model_y = RegressionWrapper(lgb.LGBMClassifier(random_state=2023, num_leaves=63, learning_rate=0.05, n_estimators=200, objective='binary'))
    model_t = lgb.LGBMClassifier(random_state=2023, num_leaves=63, learning_rate=0.05, n_estimators=200, objective='multiclass')
    est = CausalForestDML(
        model_y= model_y,
        model_t = model_t,
        discrete_treatment=True,
        max_depth=10,
        min_samples_split=10,
        cv=10,
        #,criterion='het'
        verbose=10,
        random_state=2023,
        n_estimators=100,
        subforest_size=2,
    )
    est.fit(Y.values, T=T.values,  X=X.values, W=None, cache_values=True)
    est.summary()

    
    # test model 
    print('-------------------------\n testing... \n-------------------------\n')
    cols = pd.read_csv('cols_select_test{}.txt'.format(args.cols), header=None)
    cols = list(cols[0])

    test_file_list = ['test_dataset_{}.txt'.format(i) for i in TEST_FILE_DATE]
    df_test = pd.concat([pd.read_csv(i, sep='\t') for i in test_file_list], ignore_index=True)
    
    df_test = df_test[cols]
    df_test = df_test.fillna(0)
    df_test = df_test.rename(columns={'t1.label_invert': 't1.label_purchase'})
    df_test = df_test[df_test['t1.product'].isin(list(label_encode_dict.keys()))].reset_index(drop=True)
    df_test['t1.product'] = df_test['t1.product'].apply(lambda x: label_encode_dict[x])
    print(df_test['t1.product'].value_counts())
    
    T_test, Y_test, X_test = df_test['t1.product'], df_test['t1.label_purchase'], df_test.iloc[:, :-2]
    res = est.const_marginal_effect(X_test)

    pmg_score = pmg(process_dml_result(df_test, res))
    print('test data pmg is {}\n'.format(str(pmg_score)))

    
    # save model 
    print('-------------------------\n saving model... \n-------------------------\n')
    # modelpath = 'dml_v1_1.pkl'
    modelpath = 'dml_v{}.pkl'.format(args.version)
    # joblib.dump(est, filename=modelpath)
    with open(modelpath, 'wb') as f: 
        cloudpickle.dump(est, f)
    print('model saved with path {}\n'.format(modelpath))


    # feature importance
    print('-------------------------\n generate feature importance... \n-------------------------\n')
    feature_importance_dict = {df.columns[i]: est.feature_importances_[i] for i in range(df.shape[1] - 2)}
    feature_importance = pd.DataFrame({'feature_name': df.columns[:-2], 'importance': est.feature_importances_})
    feature_importance = feature_importance.sort_values(by="importance", ascending=False)

    plt.figure(figsize=(3, 10))
    # data=feature_importance[feature_importance['importance'] > 0]
    data = feature_importance.reset_index(drop=True).loc[:50, :]
    print(data)

    
    # visualize
    fig_path = './feature_importance.png'
    fig = sns.barplot(x="importance", y="feature_name", data=data, order=data["feature_name"], orient="h")
    bar_fig = fig.get_figure()
    bar_fig.savefig(fig_path, dpi = 400)
    print('feature importance figure saved with path {}\n'.format(modelpath))
    

    

