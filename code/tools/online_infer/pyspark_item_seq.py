#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spark参数
# --conf spark.yarn.dist.archives=hdfs://xxx.tar.gz#xxx --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./xxx/ccc/bin/python --conf spark.driver.memory=20g --conf spark.driver.memoryOverhead=5g --conf spark.driver.cores=6 --conf spark.executor.memory=8g --conf spark.executor.memoryOverhead=2g --conf spark.executor.cores=4 --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=200 --conf spark.dynamicAllocation.minExecutors=50 --conf spark.sql.shuffle.partitions=1000 --conf spark.sql.merge.mode=FAST --conf spark.sql.merge.enabled=true --conf spark.sql.storeAssignmentPolicy=Legacy --conf spark.sql.legacy.timeParserPolicy=LEGACY --conf spark.sql.legacy.typeCoercion.datetimeToString.enabled=true

import joblib
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
import random

from pyspark.sql.functions import udf
from pyspark.sql.types import *


# 打分数据读取
str_sql_read = '''
select 
    uid, 
    score_item_aaa, 
    score_item_bbb, 
    score_item_ccc, 
    score_item_ddd 
from 
    xxx 
where 
    concat_ws('-', year, month, day) = '${BIZ_DATE_LINE}' 
    and exp_mode = 'xxx' 
'''

# 写表
str_sql_write = '''
insert 
    overwrite table 
        xxx 
    partition 
    (
        year='${BIZYEAR_LD}', 
        month='${BIZMONTH_LD}', 
        day='${BIZDAY_LD}', 
        role=0, 
        exp_mode='xxx'
    ) 
select 
    uid, 
    seq_item_sku, 
    seq_item_score 
from 
    table_tmp 
'''

if __name__=='__main__':
    list_item = ['aaa', 'bbb', 'ccc', 'ddd']
    dict_item_sku = {
        'aaa': ['a1', 'a2'], 
        'bbb': ['b1', 'b2', 'b3', 'b4'], 
        'ccc': ['c1'], 
        'ddd': ['d1']
    }

    bc_list_item = sc.broadcast(list_item)
    bc_dict_item_sku = sc.broadcast(dict_item_sku)

    def order(data):
        array_score = np.array([float(data['score_item_aaa']), float(data['score_item_bbb']), float(data['score_item_ccc']), float(data['score_item_ddd'])])
        list_score_rank_desc = list(array_score.argsort())[::-1]
        
        str_item_list = str('-'.join(
            list(map(lambda x: random.sample(bc_dict_item_sku.value[x], 1)[0], [bc_list_item.value[i] for i in list_score_rank_desc]))
            )
        )
        str_item_score_list = str('-'.join(['{:.10f}'.format(array_score[i]) for i in list_score_rank_desc]))
        
        return (int(data['uid']), str_item_list, str_item_score_list)

    # 打分数据读取
    df_score = spark.sql(str_sql_read)

    # 排序
    rdd_order = df_score.rdd.map(lambda x: order(x))
    df_order = spark.createDataFrame(rdd_order, ['uid', 'seq_item_sku', 'seq_item_score'])

    # 临时表
    df_order.createOrReplaceTempView('table_tmp')

    # 写表
    spark.sql(str_sql_write)
