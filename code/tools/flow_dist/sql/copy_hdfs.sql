insert 
    overwrite table 
        xxx 
    partition 
    (
        year='${yesterday_year}', 
        month='${yesterday_month}', 
        day='${yesterday_day}', 
        role=1, 
        exp_mode='xxx'
    ) 
select 
    uid, 
    group_index, 
    xxx_score_rank, 
    xxx, 
    xxx_code 
from 
    xxx 
where 
    concat_ws('-', year, month, day) = '${before_yesterday}' 
    and role = 1 
    and exp_mode = 'xxx' 
;
