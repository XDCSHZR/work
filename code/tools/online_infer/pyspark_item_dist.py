#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spark参数
# --conf spark.yarn.dist.archives=hdfs://xxx.tar.gz#xxx --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./xxx/aaa/bin/python --conf spark.driver.memory=20g --conf spark.driver.memoryOverhead=4g --conf spark.driver.cores=6 --conf spark.executor.memory=8g --conf spark.executor.memoryOverhead=2g --conf spark.executor.cores=4 --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=200 --conf spark.dynamicAllocation.minExecutors=50 --conf spark.sql.merge.mode=FAST --conf spark.sql.merge.enabled=true --conf spark.sql.storeAssignmentPolicy=Legacy --conf spark.sql.legacy.timeParserPolicy=LEGACY --conf spark.sql.legacy.typeCoercion.datetimeToString.enabled=true

import joblib
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
import random

from pyspark.sql.functions import udf
from pyspark.sql.types import *


str_sql_read = '''
select 
    uid, 
    score_item_10, 
    score_item_11, 
    score_item_12, 
    score_item_13, 
    is_xxx 
from 
    xxx 
where 
    concat_ws('-', year, month, day) = '${BIZ_DATE_LINE}' 
    and exp_mode = 'xxx' 
'''

str_sql_write = '''
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
    uid, 
    item, 
    item_id 
from 
    table_tmp 
'''

if __name__=='__main__':
    list_item = [
        'aaa', 'bbb', 'ccc', 'ddd'
    ]
    dict_item_id = {
        'aaa': 10, 
        'bbb': 11, 
        'ccc': 12, 
        'ddd': 13
    }
    
    bc_list_item = sc.broadcast(list_item)
    bc_dict_item_id = sc.broadcast(dict_item_id)
    
    def dist(data):
        # 取打分最高item进行分配
        array_score = np.array([float(data['score_item_10']), float(data['score_item_11']), float(data['score_item_12']), float(data['score_item_13'])])
        list_score_rank_desc = list(array_score.argsort())[::-1]

        score_item = array_score[list_score_rank_desc[0]]
        if score_item < 0.85 and int(data['is_xxx']) == 0:
            str_item = str('xxx')
            id_item = int(999)
        else:
            str_item = str(bc_list_item.value[list_score_rank_desc[0]])
            id_item = int(bc_dict_item_id.value[str_item])

        return (int(data['uid']), str_item, id_item)

    # 打分数据读取
    df_score = spark.sql(str_sql_read)
    
    # 分发
    rdd_order = df_score.rdd.map(lambda x: dist(x))
    df_order = spark.createDataFrame(rdd_order, ['uid', 'item', 'item_id'])

    # 临时表
    df_order.createOrReplaceTempView('table_tmp')

    # 写表
    spark.sql(str_sql_write)
