#!/bin/bash
# -*- coding: utf-8 -*-
source /etc/profile
source ~/.bash_profile

yesterday='2023-12-18'
yesterday_year=${yesterday:0:4}
yesterday_month=${yesterday:5:2}
yesterday_day=${yesterday:8:2}
echo ${yesterday_year}
echo ${yesterday_month}
echo ${yesterday_day}

python flow_dist/main_hdfs.py --yesterday $yesterday --yesterday_year $yesterday_year --yesterday_month $yesterday_month --yesterday_day $yesterday_day --config_path flow_dist/config/distribute_hdfs.yaml > flow_dist/log/dist_.log 2>&1

if [ $? == 0 ];
then
    echo 'Done successfully!'
else
    python chat.py 2>&1
    echo 'Please check log for errors!!!'
fi
