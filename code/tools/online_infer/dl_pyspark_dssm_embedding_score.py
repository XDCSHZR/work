#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spark 参数
# --conf spark.yarn.dist.archives=hdfs://xxx.tar.gz#deepctr --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./deepctr/deepctr/bin/python --conf spark.driver.memory=20g --conf spark.driver.cores=6 --conf spark.driver.memoryOverhead=5g --conf spark.executor.memory=20g --conf spark.executor.memoryOverhead=5g --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=500 --conf spark.dynamicAllocation.minExecutors=100

import joblib
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
import random

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.column import Column, _to_java_column, _to_seq


str_sql_item_embedding = '''
select 
    item_id, 
    item_embedding_0, 
    item_embedding_1, 
    item_embedding_2, 
    item_embedding_3, 
    item_embedding_4, 
    item_embedding_5, 
    item_embedding_6, 
    item_embedding_7, 
    item_embedding_8, 
    item_embedding_9, 
    item_embedding_10, 
    item_embedding_11, 
    item_embedding_12, 
    item_embedding_13, 
    item_embedding_14, 
    item_embedding_15, 
    item_embedding_16, 
    item_embedding_17, 
    item_embedding_18, 
    item_embedding_19, 
    item_embedding_20, 
    item_embedding_21, 
    item_embedding_22, 
    item_embedding_23, 
    item_embedding_24, 
    item_embedding_25, 
    item_embedding_26, 
    item_embedding_27, 
    item_embedding_28, 
    item_embedding_29, 
    item_embedding_30, 
    item_embedding_31, 
    item_embedding_32, 
    item_embedding_33, 
    item_embedding_34, 
    item_embedding_35, 
    item_embedding_36, 
    item_embedding_37, 
    item_embedding_38, 
    item_embedding_39, 
    item_embedding_40, 
    item_embedding_41, 
    item_embedding_42, 
    item_embedding_43, 
    item_embedding_44, 
    item_embedding_45, 
    item_embedding_46, 
    item_embedding_47, 
    item_embedding_48, 
    item_embedding_49, 
    item_embedding_50, 
    item_embedding_51, 
    item_embedding_52, 
    item_embedding_53, 
    item_embedding_54, 
    item_embedding_55, 
    item_embedding_56, 
    item_embedding_57, 
    item_embedding_58, 
    item_embedding_59, 
    item_embedding_60, 
    item_embedding_61, 
    item_embedding_62, 
    item_embedding_63 
from 
    xxx 
where 
    concat_ws('-', year, month, day) = '2023-03-07' 
    and exp_mode = 'xxx' 
'''

str_sql_user_embedding = '''
select 
    uid, 
    user_embedding_0, 
    user_embedding_1, 
    user_embedding_2, 
    user_embedding_3, 
    user_embedding_4, 
    user_embedding_5, 
    user_embedding_6, 
    user_embedding_7, 
    user_embedding_8, 
    user_embedding_9, 
    user_embedding_10, 
    user_embedding_11, 
    user_embedding_12, 
    user_embedding_13, 
    user_embedding_14, 
    user_embedding_15, 
    user_embedding_16, 
    user_embedding_17, 
    user_embedding_18, 
    user_embedding_19, 
    user_embedding_20, 
    user_embedding_21, 
    user_embedding_22, 
    user_embedding_23, 
    user_embedding_24, 
    user_embedding_25, 
    user_embedding_26, 
    user_embedding_27, 
    user_embedding_28, 
    user_embedding_29, 
    user_embedding_30, 
    user_embedding_31, 
    user_embedding_32, 
    user_embedding_33, 
    user_embedding_34, 
    user_embedding_35, 
    user_embedding_36, 
    user_embedding_37, 
    user_embedding_38, 
    user_embedding_39, 
    user_embedding_40, 
    user_embedding_41, 
    user_embedding_42, 
    user_embedding_43, 
    user_embedding_44, 
    user_embedding_45, 
    user_embedding_46, 
    user_embedding_47, 
    user_embedding_48, 
    user_embedding_49, 
    user_embedding_50, 
    user_embedding_51, 
    user_embedding_52, 
    user_embedding_53, 
    user_embedding_54, 
    user_embedding_55, 
    user_embedding_56, 
    user_embedding_57, 
    user_embedding_58, 
    user_embedding_59, 
    user_embedding_60, 
    user_embedding_61, 
    user_embedding_62, 
    user_embedding_63 
from 
    xxx 
where 
    concat_ws('-', year, month, day) = '${BIZ_DATE_LINE}' 
'''

# 写入结果表
str_sql_insert = '''
insert 
    overwrite table 
        xxx 
    partition 
    (
        year='${BIZYEAR_LD}', 
        month ='${BIZMONTH_LD}', 
        day='${BIZDAY_LD}', 
        exp_mode='xxx'
    ) 
select 
    uid, 
    score_item_aaa 
from 
    table_tmp 
'''

if __name__=='__main__':
    # item
    df_item_embedding = spark.sql(str_sql_item_embedding).toPandas()
    bc_df_item_embedding = sc.broadcast(df_item_embedding)
    
    list_item_id = ['score_item_{}'.format(x) for x in list(bc_df_item_embedding.value['item_id'])]
    bc_list_item_id = sc.broadcast(list_item_id)

    # user
    df_user_embedding = spark.sql(str_sql_user_embedding)
    
    list_user_embedding = [x for x in df_user_embedding.columns if x not in ['uid']]
    bc_len_user_embedding = sc.broadcast(len(list_user_embedding))

    # cosine sigmoid
    def cal_cosine_sigmoid_item(data):
        list_score = []
        
        array_uid_user_embedding = np.array(list(data)).reshape(-1, 1+bc_len_user_embedding.value)
        
        array_user_embedding = array_uid_user_embedding[:, 1:]
        array_user_embedding_norm = np.linalg.norm(array_user_embedding, axis=1)
        
        for i, row in bc_df_item_embedding.value.iterrows():
            array_item_embedding = np.tile(row.values[1:], (array_user_embedding.shape[0], 1))
            array_item_embedding_norm = np.linalg.norm(array_item_embedding, axis=1)
            
            cosine_score_top = np.sum(array_user_embedding*array_item_embedding, axis=1)
            cosine_score_bottom = array_user_embedding_norm * array_item_embedding_norm + 1e-8
            cosine_score = cosine_score_top / cosine_score_bottom
            cosine_score = np.clip(cosine_score, -1.0, 1.0)
            cosine_score_sigmoid = 1 / (1+np.exp(-cosine_score))
            
            list_score.append(float(cosine_score_sigmoid[0]))
            
        return tuple([int(array_uid_user_embedding[:, 0][0])]+list_score)
        
    rdd_score = df_user_embedding.rdd.map(lambda x: cal_cosine_sigmoid_item(x))

    df_score = spark.createDataFrame(rdd_score, ['uid']+bc_list_item_id.value)
    df_score.createOrReplaceTempView('table_tmp')

    # write
    spark.sql(str_sql_insert)
