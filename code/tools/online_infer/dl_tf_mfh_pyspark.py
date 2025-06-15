# spark 参数
# spark.executor.cores(3)
# spark.executor.memory(12G)
# spark.executor.memoryOverhead(3G)
# spark.dynamicAllocation.maxExecutors(1000)
# spark.dynamicAllocation.minExecutors(100)
# spark.yarn.appMasterEnv.PYSPARK_PYTHON(./miniconda3/bin/python3)
# spark.driver.memory(12G)
# spark.driver.cores(6)
# spark.yarn.dist.archives(hdfs://xxx/xxx.tar.gz#miniconda3)
# spark.yarn.dist.jars(hdfs://xxx/spark_tf_connector.jar)
# spark.sql.shuffle.partitions(2000)
# spark.default.parallelism(500)
# spark.driver.memoryOverhead(3G)
# spark.sql.merge.enabled(true)
# spark.sql.merge.mode(FAST)

import tensorflow as tf
import os
from pyspark import SparkFiles
from pyspark.sql.functions import struct, pandas_udf, col
import numpy as np
from pyspark.sql.column import Column, _to_java_column, _to_seq
from pyspark.sql.types import StructType, StructField, StringType, FloatType


model_name='best'
# model_name='1721812957'  # 2023.10.02~2024.06.17
# model_name='1722507582'  # increment training 2024.06.24
# model_name='1722586707'  # increment training 2024.07.08


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    value = value if isinstance(value,list) else [value]
    # if isinstance(value, type(tf.constant(0))):
    #     value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    value = value if isinstance(value,list) else [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    value = value if isinstance(value,list) else [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# 解决模型不能序列化的问题 加个wrapper
class EstimatorWrapper(object):
    def __init__(self, model_path, tf_version):
        self.estimator = self.load_model(model_path, tf_version)
        self.model_path = model_path
        self.tf_version = tf_version

    def __getstate__(self):
        return "{}\t{}".format(self.model_path, self.tf_version)

    def __setstate__(self, path_and_version):
        model_path, tf_version = path_and_version.split('\t')
        self.estimator = self.load_model(model_path, tf_version)
        self.model_path = model_path
        self.tf_version = tf_version

    def load_model(self, model_path, tf_version):
        return tf.saved_model.load(model_path)


def process_partition(row):
    if row == []:
        return []

    loaded_model = tf.saved_model.load(SparkFiles.get(model_name))
    predict_fn = loaded_model.signatures["serving_default"]

    result = predict_fn(examples=tf.constant(row))
    uid = result['uid'].numpy().astype(int)
    score_id_1 = result['pred_id_1'].numpy().astype(float)
    score_id_2 = result['pred_id_2'].numpy().astype(float)
    score_id_3 = result['pred_id_3'].numpy().astype(float)
    cancat_np=np.concatenate((uid, score_id_1, score_id_2, score_id_3), axis=1).tolist()

    return cancat_np


def serialize_example(data):
    res = []
    for row in data:
        feature = {}
        for key, value in row.items():
            if value is None:
                continue
            elif isinstance(value,list) :
                tmp = []
                for item in value:
                    tmp.append(item.encode('utf-8'))
                feature[key] = _bytes_feature(tmp)
            elif isinstance(value, str) :
                feature[key] = _bytes_feature(value.encode('utf-8'))
            elif isinstance(value, float):
                feature[key] = _float_feature(value)
            elif isinstance(value, int):
                feature[key] = _int64_feature(value)

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        res.append(example_proto.SerializeToString())
    return [res]


def start(spark, input, output):
    path='hdfs://xxx/'+model_name+'/'

    spark.sparkContext.addFile(path, recursive=True)

    query = '''
    select 
        cast(t_1.uid as string) as uid, 
        t_1.obs_dt as obs_dt, 
        t_1.features_date as features_date, 
        0 as domain_id, 
        0 as id_1, 
        0 as id_2, 
        0 as id_3, 
        0 as label_1, 
        0 as label_2, 
        0 as label_3, 
        0 as label_4, 
        
    from 
    (
        select 
            t.uid as uid, 
            t.obs_dt as obs_dt, 
            t.features_date as features_date, 
            
        from 
        (
            select 
                uid, 
                concat_ws('-', year, month, day) as obs_dt, 
                features_date, 
                row_number() over(partition by uid, concat_ws('-', year, month, day), features_date order by 1 desc) as rn, 
                
            from 
                xxx.xxx 
            where 
                concat_ws('-', year, month, day) = '' 
        ) as t 
        where 
            t.rn = 1 
    ) as t_1 
    left join 
    (
        select 
            t.uid as uid, 
            
        from 
        (
            select 
                uid, 
                row_number() over(partition by uid, concat_ws('-', year, month, day), features_date order by 1 desc) as rn, 
                
            from 
                xxx.xxx 
            where 
                concat_ws('-', year, month, day) = '' 
        ) as t 
        where 
            t.rn = 1 
    ) as t_2 
    on t_1.uid = t_2.uid 
    left join 
    (
        select 
            t.uid as uid, 
            
        from 
        (
            select 
                uid, 
                row_number() over(partition by uid, concat_ws('-', year, month, day), features_date order by 1 desc) as rn, 
                
            from 
                xxx.xxx 
            where 
                concat_ws('-', year, month, day) = '' 
        ) as t 
        where 
            t.rn = 1 
    ) as t_3 
    on t_1.uid = t_3.uid 
    '''
    df = spark.sql(query)
    pred_columns = ['uid', 'score_id_1', 'score_id_2', 'score_id_3']
    columns = df.columns

    rdd_of_dicts = df.repartition(10000).rdd.map(lambda row: {col: row[col] for col in columns})
    res = rdd_of_dicts.mapPartitions(serialize_example).flatMap(process_partition)
    res = res.toDF(pred_columns)
    res.createOrReplaceTempView('tmp')
    
    this_sql_1 = '''
    insert 
        overwrite table 
            xxx.xxx 
        partition 
        (
            year='', 
            month='', 
            day='', 
            exp_mode=''
        ) 
    select 
        uid, 
        score_id_1, 
        score_id_2, 
        score_id_3 
    from 
        tmp 
    '''
    spark.sql(this_sql_1)