#!/bin/bash

exp_path="hdfs://xxx/year="${BIZYEAR_LD}"/month="${BIZMONTH_LD}"/day="${BIZDAY_LD}"/exp_mode=xxx"
echo $exp_path

wait_nums=48

i=1
j=1
flag=0
echo "check for table..."
while [ $i -le $wait_nums ]
do
    hadoop fs -test -e $exp_path
    if [ $? -eq 0 ] ;then 
        flag=1
        echo 'uploading...wait for 30 mins...'
        sleep 30m
        break
    else
        echo 'Wait for table, time: '$i 
        sleep 30m
        let i++
    fi
done

if [ $flag -eq 0 ] ;then 
    echo 'Error! table is not exist!'
    exit
else 
    echo 'table exists, path is '$exp_path
fi
