load data local inpath 
    '${hiveconf:upload_file}' 
overwrite into table 
    xxx 
partition 
(
    year = '${hiveconf:yesterday_year}', 
    month = '${hiveconf:yesterday_month}', 
    day = '${hiveconf:yesterday_day}', 
    role = 0, 
    exp_mode = 'xxx'
)
;
