#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spark参数
# --conf spark.yarn.dist.archives=hdfs://xxx.tar.gz#xxx --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./xxx/ccc/bin/python --conf spark.driver.memory=20g --conf spark.driver.memoryOverhead=5g --conf spark.driver.cores=6 --conf spark.executor.memory=12g --conf spark.executor.memoryOverhead=4g --conf spark.executor.cores=4 --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.maxExecutors=200 --conf spark.dynamicAllocation.minExecutors=50 --conf spark.sql.shuffle.partitions=1000 --conf spark.sql.merge.mode=FAST --conf spark.sql.merge.enabled=true --conf spark.sql.storeAssignmentPolicy=Legacy --conf spark.sql.legacy.timeParserPolicy=LEGACY --conf spark.sql.legacy.typeCoercion.datetimeToString.enabled=true

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
    group_index, 
    score_item_01, 
    score_item_02 
from 
    xxx 
where 
    concat_ws('-', year, month, day) = '${BIZ_DATE_LINE}' 
    and role = 1 
    and exp_mode = 'xxx' 
'''

if __name__=='__main__':
    df_rule_score = spark.sql(str_sql_read)

    df_rule_score.repartition(1).write.mode('overwrite').format('csv').\
        option('encoding', 'utf-8').option('header', True).option('delimiter', '\t').\
        save('hdfs://'+'xxx/year=${BIZYEAR_LD}/month=${BIZMONTH_LD}/day=${BIZDAY_LD}/role=1/'+'xxx.csv')
