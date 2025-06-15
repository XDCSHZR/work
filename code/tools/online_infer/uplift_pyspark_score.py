#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spark 参数
# --conf spark.yarn.dist.archives=hdfs://xxx.tar.gz#xxx --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./xxx/xxx/bin/python --conf spark.driver.memory=20g --conf spark.executor.memory=12g --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=500 --conf spark.dynamicAllocation.minExecutors=100 --files no_secret_bt_20220206_20220227_union_XLearnerLgb.model --conf spark.hadoop.mapred.max.split.size=26214400 --conf spark.driver.cores=6 --conf spark.executor.memoryOverhead=2g --conf spark.driver.memoryOverhead=5g --conf spark.reducer.maxReqsInFlight=10 --conf spark.reducer.maxBlocksInFlightPerAddress=10 --conf spark.task.cpus=4

import joblib
import causalml
import pyspark.sql.functions as F
import pandas as pd
import numpy as np

from pyspark.sql.functions import udf
from pyspark.sql.types import *

# 模型
learner_x_lgb = joblib.load('xxx.model')
bc_learner_x_lgb = sc.broadcast(learner_x_lgb)

# 特征数据读取
str_sql_read = '''
select 
    * 
from 
    xxx 
where 
    concat_ws('-', year, month, day) = '{ymd}' 
'''.format(ymd='${BIZ_DATE_LINE}')
df_test = spark.sql(str_sql_read)

features = list(df_test.columns)
bc_features = sc.broadcast(features[1:-3])

dict_p = {
    'treatment_1': np.array([0.0653]), 
    'treatment_2': np.array([0.2908]), 
    'treatment_3': np.array([0.0653]), 
    'treatment_4': np.array([0.0476]), 
    'treatment_5': np.array([0.0367])
}
bc_dict_p = sc.broadcast(dict_p)

def prdeictBatch(datas):
    for data in datas:
        tmp = np.array(list(data)).reshape(-1, 1+len(bc_features.value)+3)[:, 1:-3]
        score = bc_learner_x_lgb.value.predict(tmp, p=bc_dict_p.value)
        score_coupon_20 = float(score[:, 0])
        score_coupon_50 = float(score[:, 1])
        score_coupon_100 = float(score[:, 2])
        score_coupon_150 = float(score[:, 3])
        score_coupon_180 = float(score[:, 4])
        
        try:
            yield (int(data['uid']), score_coupon_20, score_coupon_50, score_coupon_100, score_coupon_150, score_coupon_180)
        except StopIteration:
            return

rdd_pred = df_test.repartition(5000).rdd.mapPartitions(lambda x: prdeictBatch(x))
df_pred = spark.createDataFrame(rdd_pred, ['uid', 'score_coupon_20', 'score_coupon_50', 'score_coupon_100', 'score_coupon_150', 'score_coupon_180'])

# 临时表
df_pred.createOrReplaceTempView('table_tmp')

# 写表
str_sql_write = '''
insert 
    overwrite table 
        xxx 
    partition 
    (
        year='{y}', 
        month ='{m}', 
        day='{d}'
    ) 
select 
    uid, 
    score_coupon_20, 
    score_coupon_50, 
    score_coupon_100, 
    score_coupon_150, 
    score_coupon_180 
from 
    table_tmp 
'''.format(y='${BIZYEAR_LD}', m='${BIZMONTH_LD}', d='${BIZDAY_LD}')
spark.sql(str_sql_write)
