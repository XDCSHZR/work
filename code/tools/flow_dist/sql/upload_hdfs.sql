load data local inpath 
    '${upload_file}' 
overwrite into table 
    xxx 
partition 
(
    year = '${yesterday_year}', 
    month = '${yesterday_month}', 
    day = '${yesterday_day}', 
    role = 1, 
    exp_mode = 'xxx'
)
;
