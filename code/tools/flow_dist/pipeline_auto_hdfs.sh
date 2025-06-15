#!/bin/bash
# -*- coding: utf-8 -*-
source /etc/profile
source ~/.bash_profile

today=$(date -d '-0 day' '+%Y-%m-%d') # 今天
echo ${today}

yesterday=$(date -d "$today -1 day" '+%Y-%m-%d') # 昨天
echo ${yesterday}

yesterday_year=${yesterday:0:4}
yesterday_month=${yesterday:5:2}
yesterday_day=${yesterday:8:2}
echo ${yesterday_year}
echo ${yesterday_month}
echo ${yesterday_day}

yesterday_weekday=$(date -d $yesterday +%w) # 昨天周几
if [ $yesterday_weekday == 0 ]; # 如果周天替换为前一天周六
then
    echo 'sunday!!!'
    
    before_yesterday=$(date -d "$today -2 day" '+%Y-%m-%d')
    echo ${before_yesterday}
    
    spark-sql --queue xxx -d yesterday_year=${yesterday_year} -d yesterday_month=${yesterday_month} -d yesterday_day=${yesterday_day} -d before_yesterday=${before_yesterday} -f flow_dist/sql/copy.sql > flow_dist/log/dist_${today}.log 2>&1
    
    if [ $? == 0 ];
    then
        echo 'Copy done!'
        python flow_dist/chat_done.py --config_path flow_dist/config/distribute_hdfs.yaml 2>&1
    fi
else
    python flow_dist/main_hdfs.py --yesterday $yesterday --yesterday_year $yesterday_year --yesterday_month $yesterday_month --yesterday_day $yesterday_day --config_path flow_dist/config/distribute_hdfs.yaml > flow_dist/log/dist_${today}.log 2>&1
fi

if [ $? == 0 ];
then
    echo 'Done successfully!'
else
    python flow_dist/chat.py 2>&1
    echo 'Please check log for errors!!!'
fi
