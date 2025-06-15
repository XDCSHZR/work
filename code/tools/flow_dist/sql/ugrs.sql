select 
    t_id.uid as uid, 
    t_id.group_index as group_index, 
    t_rule.rule_01 as rule_01, 
    t_rule.rule_02 as rule_02, 
    t_rule.rule_04 as rule_04, 
    t_rule.rule_06 as rule_06, 
    t_rule.rule_03 as rule_03, 
    t_score_1.score_01 * t_score_2.score_01 as score_01, 
    t_score_1.score_02 * t_score_2.score_02 as score_02, 
    t_score_1.score_03 * t_score_2.score_03 as score_03, 
    t_score_1.score_04 * t_score_2.score_04 as score_04, 
    t_score_1.score_10 * t_score_2.score_10 as score_10 
from 
(
    select 
        uid, 
        group_index 
    from 
        xxx 
    where 
        concat_ws('-', year, month, day) = '${hiveconf:dt}' 
        and group_index in ${hiveconf:g} 
) as t_id 
left join 
(
    select 
        t_rule_0.uid as uid, 
        t_rule_0.rule_01 as rule_01, 
        t_rule_0.rule_02 & t_rule_1.rule_02 as rule_02, 
        t_rule_0.rule_03 as rule_03, 
        t_rule_0.rule_04 as rule_04, 
        t_rule_0.rule_06 as rule_06 
    from 
    (
        select 
            uid, 
            rule_01, 
            rule_02, 
            rule_03, 
            rule_04, 
            rule_06 
        from 
            xxx 
        where 
            concat_ws('-', year, month, day) = '${hiveconf:dt}' 
            and role = 0 
    ) as t_rule_0 
    left join 
    (
        select 
            uid, 
            rule_02 
        from 
            xxx 
        where 
            concat_ws('-', year, month, day) = '${hiveconf:dt}' 
            and role = 0 
    ) as t_rule_1 
    on t_rule_0.uid = t_rule_1.uid 
) as t_rule 
on t_id.uid = t_rule.uid 
left join 
(
    select 
        uid, 
        score_01, 
        score_02, 
        score_03, 
        score_04, 
        score_10 
    from 
        xxx 
    where 
        concat_ws('-', year, month, day) = '${hiveconf:dt}' 
        and role = 0 
        and exp_mode = 'xxx' 
) as t_score_1 
on t_id.uid = t_score_1.uid 
left join 
(
    select 
        uid, 
        score_01, 
        score_02, 
        score_03, 
        score_04, 
        score_10 
    from 
        xxx 
    where 
        concat_ws('-', year, month, day) = '${hiveconf:dt}' 
        and role = 0 
        and exp_mode = 'xxx' 
) as t_score_2 
on t_id.uid = t_score_2.uid 
;
