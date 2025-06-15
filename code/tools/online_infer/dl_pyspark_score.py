#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spark参数
# --conf spark.yarn.dist.archives=hdfs://xxx.tgz#python3 --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./python3/bin/python3 --conf spark.driver.memory=20g --conf spark.driver.memoryOverhead=5g --conf spark.driver.cores=6 --conf spark.executor.memory=25g --conf spark.executor.memoryOverhead=5g --conf spark.executor.cores=4 --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=2000 --conf spark.dynamicAllocation.minExecutors=600 --conf spark.sql.shuffle.partitions=10000 --conf spark.sql.merge.mode=FAST --conf spark.sql.merge.enabled=true --files hdfs://mmoe_b6t2.pt

spark.conf.set('spark.sql.execution.arrow.enabled', 'true')
spark.conf.set('spark.sql.execution.arrow.maxRecordsPerBatch', '10000')

import pyspark.sql.functions as F
import pandas as pd
import numpy as np
import torch
import joblib
import random

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.column import Column, _to_java_column, _to_seq


str_sql_feats = '''
select 
    -- id
    uid, 
    -- 特征
    -- sparse
    concat_ws(',', 
        cast(xxx as string), 
        cast(xxx as string)
    ) as sparse_features, 
    -- dense
    concat_ws(',', 
        cast(xxx as string), 
        cast(xxx as string)
    ) as dense_features 
from 
    xxx 
where 
    concat_ws('-', year, month, day) = '${BIZ_DATE_LINE}' 
    and exp_mode = 'xxx' 
'''

# 写入结果表
insert_sql = '''
insert 
    overwrite table 
        xxx 
    partition 
    (
        year='${BIZYEAR_LD}', 
        month='${BIZMONTH_LD}', 
        day='${BIZDAY_LD}', 
        role=2, 
        exp_mode='xxx'
    ) 
select 
    uid, 
    -- 一般负采样打分修正公式
    score_a / (score_a+(1-score_a)/1.0) as score_01, 
    -- 链路负采样打分修正公式
    -- 假设label负样本（前链路）在label_负样本（后链路）中的占比为定值
    -- r = label负样本 / label_负样本
    -- s_ = s / (s+(1-s)*(1-r)+(1-s)*r/label负样本采样比例)
    score_b / (score_b+(1-score_b)*(1-0.7717)+(1-score_b)*0.7717/0.5676) as score_02 
from 
    table_tmp 
'''

if __name__=='__main__':
    # 模型&参数&广播
    MODEL_NAME = 'mmoe_b6t2.pt'
    NUM_item_TABLE = 7
    NUM_TASK = 5
    NUM_PARTITIONS = 40000
    item_table_cols = [
        'item_0_table_2', 'item_0_table_1'
    ]
    feature_cols = ['{}_feats'.format(item_table_cols[i]) for i in range(NUM_item_TABLE)]
    score_cols = ['{}_scores'.format(item_table_cols[i]) for i in range(NUM_item_TABLE)]
    score_final_cols = [
        'score_a', 'score_b'
    ]
    
    bc_MODEL_NAME = sc.broadcast(MODEL_NAME)
    bc_NUM_TASK = sc.broadcast(NUM_TASK)
    bc_NUM_PARTITIONS = sc.broadcast(NUM_PARTITIONS)
    bc_item_table_cols = sc.broadcast(item_table_cols)
    bc_feature_cols = sc.broadcast(feature_cols)
    bc_score_cols = sc.broadcast(score_cols)
    bc_score_final_cols = sc.broadcast(score_final_cols)
    
    # batch打分udf
    def get_model_for_eval():
        model = torch.jit.load(bc_MODEL_NAME.value, map_location='cpu')
        return model

    @pandas_udf(FloatType())
    def udf_predict_batch(arr: pd.Series) -> pd.Series:
        model = get_model_for_eval() # .pt模型无法序列化广播，在udf中调取
        
        arr = np.vstack(arr.map(lambda x: np.float32(x.split(','))).values)
        arr = torch.tensor(arr)
        with torch.no_grad():
            predictions = list(torch.sigmoid(model(arr).type(torch.float64)).cpu().numpy()[:, bc_NUM_TASK.value-1])
            
        return pd.Series(predictions)
        
    # 特征读取（已处理）
    df = spark.sql(str_sql_feats)
    
    # 打分
    df = df.select(
        ['*'] 
        + [
            lit(','.join(['1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0'])).alias(bc_item_table_cols.value[0]), 
            lit(','.join(['1.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', '0.0'])).alias(bc_item_table_cols.value[1])
          ]
    )
    df = df.select(
        ['uid'] 
        + [
            concat_ws(',', col(bc_item_table_cols.value[0]), col('sparse_features'), col('dense_features')).alias(bc_feature_cols.value[0]), 
            concat_ws(',', col(bc_item_table_cols.value[1]), col('sparse_features'), col('dense_features')).alias(bc_feature_cols.value[1])
          ]
    )
    
    df = df.repartition(bc_NUM_PARTITIONS.value)
    df = df.select(
        ['uid'] 
        + [
            udf_predict_batch_pass_1(bc_feature_cols.value[0]).alias(bc_score_cols.value[0]), 
            udf_predict_batch_pre_pass(bc_feature_cols.value[1]).alias(bc_score_cols.value[1])
          ]
    )
    
    df = df.select(
        ['uid'] 
        + [
            col(bc_score_cols.value[0]).cast('double').alias(bc_score_final_cols.value[0]), 
            col(bc_score_cols.value[1]).cast('double').alias(bc_score_final_cols.value[1])
          ]
    )
    df.createOrReplaceTempView('table_tmp')

    # 写表
    spark.sql(insert_sql)
