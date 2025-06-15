#!/bin/bash
# -*- coding: utf-8 -*-
source /etc/profile
source ~/.bash_profile

today=$(date -d '-0 day' '+%Y-%m-%d') # 今天
#today_year=${today:0:4}
#today_month=${today:5:2}
#today_day=${today:8:2}
#echo ${today_year}
#echo ${today_month}
#echo ${today_day}

yesterday=$(date -d "$today -1 day" '+%Y-%m-%d') # 昨天
yesterday_year=${yesterday:0:4}
yesterday_month=${yesterday:5:2}
yesterday_day=${yesterday:8:2}
echo ${yesterday_year}
echo ${yesterday_month}
echo ${yesterday_day}

.conda/envs/xxx/bin/python main.py --yesterday $yesterday --yesterday_year $yesterday_year --yesterday_month $yesterday_month --yesterday_day $yesterday_day --config_path flow_dist/config/distribute.yaml > flow_dist/log/dist_${today}.log 2>&1

if [ $? == 0 ];
then
    echo 'Done successfully!'
else
    .conda/envs/xxx/bin/python chat.py 2>&1
    echo 'Please check log for errors!!!'
fi
