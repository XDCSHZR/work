-- spark参数
--conf spark.sql.merge.mode=FAST --conf spark.sql.merge.enabled=true --conf spark.driver.memory=20g --conf spark.driver.memoryOverhead=4g --conf spark.driver.cores=6 --conf spark.executor.memory=25G --conf spark.executor.memoryOverhead=5g --conf spark.executor.cores=4 --conf spark.dynamicAllocation.maxExecutors=2000 --conf spark.dynamicAllocation.minExecutors=500 --conf spark.sql.shuffle.partitions=10000 --conf spark.rss.enabled=true --conf spark.sql.storeAssignmentPolicy=Legacy --conf spark.sql.legacy.timeParserPolicy=LEGACY --conf spark.sql.legacy.typeCoercion.datetimeToString.enabled=true

add jar hdfs://xgbudf-1.0-SNAPSHOT.jar;
add file hdfs://xxx.model;
create temporary function get_score as 'PushPrediction';

insert 
    overwrite table 
        xxx 
    partition 
    (
        year='${BIZYEAR_LD}', 
        month='${BIZMONTH_LD}', 
        day='${BIZDAY_LD}', 
        exp_mode='xxx'
    ) 
select 
    t_score.uid as uid, 
    t_score.score_aaa_before / (t_score.score_aaa_before+(1-t_score.score_aaa_before)/1.0) as score_aaa, 
    t_score.score_bbb_before / (t_score.score_bbb_before+(1-t_score.score_bbb_before)/1.0) as score_bbb 
from 
(
    select 
        t_concat.uid as uid, 
        get_score(concat_ws(',', 
            '1.0', '0.0', 
            t_concat.feature_str), 'xxx.model'
        ) as score_aaa_before, 
        get_score(concat_ws(',', 
            '0.0', '1.0', 
            t_concat.feature_str), 'xxx.model'
        ) as score_bbb_before 
    from 
    (
        select 
            t_feature.uid as uid, 
            concat_ws(',', 
                cast(t_feature.aaa as string), 
                cast(t_feature.bbb as string)
            ) as feature_str 
        from 
        (
            select 
                t_base_people.uid as uid, 
                -- feature
                coalesce(t_xxx.aaa, 0) as aaa, 
                coalesce(t_xxx.bbb, -1) as bbb 
            from 
            (
                select 
                    uid 
                from 
                    xxx 
                where 
                    concat_ws('-', year, month, day) = '${BIZ_DATE_LINE}' 
            ) as t_base_people 
            left join 
            (
                select 
                    uid, 
                    aaa, 
                    bbb 
                from 
                    xxx 
                where 
                    concat_ws('-', year, month, day) = '${LW_END_DATE}' 
            ) as t_xxx 
            on t_base_people.uid = t_xxx.uid 
        ) as t_feature 
    ) as t_concat 
) as t_score 
;
