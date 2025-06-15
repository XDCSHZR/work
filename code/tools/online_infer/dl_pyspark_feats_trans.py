#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spark参数
# --conf spark.yarn.dist.archives=hdfs://xxx.tar.gz#deepctr --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./deepctr/deepctr/bin/python --conf spark.driver.memory=20g --conf spark.driver.memoryOverhead=5g --conf spark.driver.cores=6 --conf spark.executor.memory=25g --conf spark.executor.memoryOverhead=5g --conf spark.executor.cores=4 --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=1000 --conf spark.dynamicAllocation.minExecutors=250 --conf spark.sql.shuffle.partitions=4000 --files ss.pickle,dict_lbe.pickle

import joblib
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
import random

from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql.column import Column, _to_java_column, _to_seq
from sklearn.preprocessing import StandardScaler, LabelEncoder


# import atexit,os,platform,warnings,py4j
# from pyspark import SparkConf
# from pyspark.context import SparkContext
# from pyspark.sql import SparkSession, SQLContext
# SparkContext._ensure_initialized()
# spark = SparkSession._create_shell_session()
# sc = spark.sparkContext
# sql = spark.sql
# sqlContext = spark._wrapped
# sqlCtx = sqlContext

str_sql_feats = '''
select 
    -- id
    uid, 
    -- 特征
    -- sparse
    xxx, 
    -- dense
    xxx 
from 
    xxx 
where 
    concat_ws('-', year, month, day) = '${BIZ_DATE_LINE}' 
    and exp_mode = 'xxx' 
'''

str_sql_feats_trans = '''
insert 
    overwrite table 
        xxx 
    partition 
    (
        year='${BIZYEAR_LD}', 
        month='${BIZMONTH_LD}', 
        day='${BIZDAY_LD}', 
        exp_mode='xxx'
    ) 
select 
    -- id
    uid, 
    -- 特征
    -- sparse
    xxx, 
    -- dense
    xxx 
from 
    table_tmp 
'''

if __name__=='__main__':
    # 原始特征
    df = spark.sql(str_sql_feats)
    
    # 离散&连续特征
    list_sparse_feats = [
        xxx
    ]
    list_dense_feats = [x for x in df.columns if x not in ['uid']+list_sparse_feats]

    bc_list_sparse_feats = sc.broadcast(list_sparse_feats)
    bc_list_dense_feats = sc.broadcast(list_dense_feats)
    
    # 数据预处理模型（离散labelEncode，连续标准归一化）
    dict_lbe = joblib.load('./dict_lbe.pickle')
    bc_dict_lbe = sc.broadcast(dict_lbe)
    
    ss = joblib.load('./ss.pickle')
    bc_ss = sc.broadcast(ss)
    
    # 数据预处理
    def transform(data):
        array_X = np.array(list(data)).reshape(-1, 1+len(bc_list_sparse_feats.value)+len(bc_list_dense_feats.value))
        
        array_X_sparse = array_X[:, 1:(1+len(bc_list_sparse_feats.value))]
        list_X_sparse_trans = [int(0) if array_X_sparse[0][i] not in bc_dict_lbe.value[x].classes_ else int(bc_dict_lbe.value[x].transform([array_X_sparse[0][i]])[0]) 
                               for i, x in enumerate(bc_list_sparse_feats.value)]
        
        array_X_dense = array_X[:, (1+len(bc_list_sparse_feats.value)):]
        array_X_dense_ss = bc_ss.value.transform(array_X_dense)
        list_X_dense_ss = [float(x) for x in array_X_dense_ss[0]]
        
        return tuple([int(array_X[:, 0][0])]+list_X_sparse_trans+list_X_dense_ss)
        
    rdd_X_transform = df.repartition(10000).rdd.map(lambda x: transform(x))
    
    df_trans = spark.createDataFrame(rdd_X_transform, ['uid']+bc_list_sparse_feats.value+bc_list_dense_feats.value)
    df_trans.createOrReplaceTempView('table_tmp')
    
    # 写表
    spark.sql(str_sql_feats_trans)
