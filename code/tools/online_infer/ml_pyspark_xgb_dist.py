#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spark 参数
# --files sparkxgb1.zip --jars xgboost4j-spark-0.90.jar,xgboost4j-0.90.jar --conf spark.yarn.dist.archives=hdfs://xxx.tgz#python3.6.9 --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./python3.6.9/bin/python3 --conf spark.driver.cores=4 --conf spark.executor.cores=4 --conf spark.driver.memoryOverhead=4096 --conf spark.sql.shuffle.partitions=4000 --conf spark.executor.memory=16g --conf spark.default.parallelism=500 --num-executors 99 --conf spark.executor.memoryOverhead=3072

import pyspark
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import col
from pyspark.sql import SparkSession,Row
from pyspark.sql.types import Row,StructField,StructType,StringType,IntegerType,DoubleType
import pandas as pd
import random
import sys
import pyspark.sql.functions as F

spark.sparkContext.addPyFile("sparkxgb1.zip")
from sparkxgb import XGBoostClassifier


string_feas = [
    xxx
]
objective = 'binary:logistic'
maxDepth = 5
eta = 0.05
subsample = 0.9
colsampleBytree = 0.9
numRound = 500
early_stopping_rounds = 50

def evaluate(model,testDF):
    pred = model.transform(testDF)
    predictionAndLabels = pred.rdd.map(lambda x :(x.label,float(x.probability[1])))
    # Instantiate metrics object
    metrics = BinaryClassificationMetrics(predictionAndLabels)
    # Area under precision-recall curve
    print("Area under PR = %s" % metrics.areaUnderPR)
    # Area under ROC curve
    print("Area under ROC = %s" % metrics.areaUnderROC)

def get_feature_importance(model,f_list):
    """
    解析特征重要性
    :param model:
    :param f_list:
    :return:
    """
    df_importance = pd.DataFrame({'feature_name': f_list, 'idx': range(len(f_list))})

    FeatureScoreMap = model.stages[1].nativeBooster.getScore("", "gain") # 注意xgboost是在pipline的第几个步骤
    file = open('importance.csv', "w+")
    print(FeatureScoreMap, file=file)
    file.close()
    f1 = open('importance.csv')
    line = f1.readline()
    data = line.replace(',', '\n').replace('->', ',').replace('Map(', '').replace(')', '').replace('f', '')
    file = open('importance.csv', "w+")
    print(data, file=file)
    file.close()
    df_temp = pd.read_csv('importance.csv', header=None, names=["idx", "weight"])
    df_importance = df_importance.merge(df_temp, on="idx")
    df_importance.sort_values(by=['weight'], ascending=False, inplace=True)
    print(df_importance)
    return df_importance

if __name__ == '__main__':
    SavePath = 'hdfs://'

    df = spark.read.parquet(SavePath)
    print('data cols: ', df.columns)
    # df = df.select(*(col(c).cast("float").alias(c) for c in df.columns))
    # df = df.na.fill(-9.0)
    # f_list = df.columns[6:]

    f_list = list()
    for fea in df.columns[6:]:
        if fea in string_feas:
            continue
        f_list.append(fea)

    print('feas cols: ', f_list)
    trainDF, testDF = df.randomSplit([0.9, 0.1], seed=22)
    vectorAssembler = VectorAssembler().setInputCols(f_list).setHandleInvalid("skip").setOutputCol("features")

    labels_list = ['aaa', 'bbb']

    all_labels_actual_imp_df, all_labels_actual_imp_dict = pd.DataFrame(), dict()
    # step1 第一遍特征重要度筛选
    for label in labels_list:
        print('train label [{}] ...'.format(label))
        xgboost=XGBoostClassifier(featuresCol="features"
                                    , labelCol=label, objective='binary:logistic'
                                    , maxDepth=maxDepth, eta=eta, subsample=subsample
                                    , colsampleBytree=colsampleBytree, missing=0.0
                                    , numRound=numRound, numWorkers=500, nthread=1)

        pipeline = Pipeline(stages=[vectorAssembler,xgboost])
        model = pipeline.fit(trainDF)
        evaluate(model, testDF)
        model_save_path = 'hdfs://spark_xgb_{}_{}{}{}/'\
                            .format(label, '${V_PARYEAR}', '${V_PARMONTH}', '${V_PARDAY}')
        model.write().overwrite().save(model_save_path)
        print('model save success to [{}]'.format(model_save_path))
        
        feas_imp = get_feature_importance(model,f_list)
        all_labels_actual_imp_dict[label] = feas_imp # label 对应其特征重要度 
        all_labels_actual_imp_df = pd.concat([actual_imp_df, feas_imp], axis=0) # 合并所有label的重要特征进行下一步的筛选
        feas_imp = spark.createDataFrame(feas_imp)
        file = 'hdfs://spark_xgb_{}_imp_{}{}{}.csv'\
                            .format(label, '${V_PARYEAR}', '${V_PARMONTH}', '${V_PARDAY}')
        feas_imp.coalesce(1).write.csv(path=file, header=True, sep=",", mode='overwrite')
        print('feas imp save success to [{}]'.format(feas_imp))
        print('='*50)

    # step2 基于step1的初步筛选结果使用null-importance 进一步筛选
    RandomCnt = 10
    f_list = list(set(all_labels_actual_imp_df['feature_name']))
    for label in labels_list:
        print('train bad label [{}]'.format(label))
        bad_imp_df = pd.DataFrame()
        for i in range(RandomCnt):
            # 打乱label
            bad_label = 'bad_{}_label'.format(label)
            df = df.withColumn('rand', F.rand(seed=random.randint(0,100)))
            tmp = df.select(label).withColumn('rand', F.rand(seed=random.randint(0,100)))
            tmp = tmp.select(label).orderBy(tmp.rand)
            df = df.withColumn(bad_label, tmp[label])

            trainDF, testDF = df.randomSplit([0.9, 0.1])
            vectorAssembler = VectorAssembler().setInputCols(f_list).setHandleInvalid("skip").setOutputCol("features")

            print('train label [{} -> {}] ...'.format(bad_label, i))
            xgboost=XGBoostClassifier(featuresCol="features"
                                        , labelCol=bad_label, objective='binary:logistic'
                                        , maxDepth=maxDepth, eta=eta, subsample=subsample
                                        , colsampleBytree=colsampleBytree, missing=0.0
                                        , numRound=numRound, numWorkers=500, nthread=1)

            pipeline = Pipeline(stages=[vectorAssembler,xgboost])
            model = pipeline.fit(trainDF)

            feas_imp = get_feature_importance(model,f_list)
            bad_imp_df = pd.concat([bad_imp_df, feas_imp], axis=0) # 合并所有label的重要特征进行下一步的筛选
            feas_imp = spark.createDataFrame(feas_imp)
            file = 'hdfs://spark_xgb_bad_{}_{}_imp_{}{}{}.csv'\
                                .format(bad_label, i, '${V_PARYEAR}', '${V_PARMONTH}', '${V_PARDAY}')
            feas_imp.coalesce(1).write.csv(path=file, header=True, sep=",", mode='overwrite')
            print('feas imp save success to [{}]'.format(feas_imp))
            df = df.drop('rand')
            print('='*50)

        feature_scores = []
        for _f in all_labels_actual_imp_dict[label].unique():
            actual_imp_df = all_labels_actual_imp_dict[label]
            f_null_imps_gain = bad_imp_df.loc[bad_imp_df['feature_name'] == _f, 'weight'].values
            f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature_name'] == _f, 'weight'].mean()
            gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero

            feature_scores.append((_f, gain_score))
        scores_df = pd.DataFrame(feature_scores, columns=['feature_name', 'split_score', 'gain_score'])
        scores_df = spark.createDataFrame(scores_df)
        file = 'hdfs://spark_xgb_final_{}_imp_{}{}{}.csv'\
                            .format(label, '${V_PARYEAR}', '${V_PARMONTH}', '${V_PARDAY}')
        scores_df.coalesce(1).write.csv(path=file, header=True, sep=",", mode='overwrite')
        print('{} label final imp feas file: {}'.format(label, file))
        print('='*50)
