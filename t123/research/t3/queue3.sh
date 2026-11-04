#!/bin/bash
cd "$(dirname "$0")/../../.."
L=t123/research/t3/logs
run(){ n=$1; shift; python -u t123/research/t3/$@ 2>&1 | grep -E --line-buffered "fold|RESULT|Error|Traceback|AUC" > $L/$n.log; }
run clf_s1  exp_lgb_hurdle.py clf_s1 --clf_only --seed 1
run clf_s2  exp_lgb_hurdle.py clf_s2 --clf_only --seed 2
python t123/research/t3/avg_q.py pavg_d5 --q xq_d5 --p lgbh_log,clf_s1,clf_s2 > $L/pavg_d5.log 2>&1
run xq_d5s1 exp_xgbq.py xq_d5s1 --offset prev --extra --depth 5 --seed 1
run xq_d5s2 exp_xgbq.py xq_d5s2 --offset prev --extra --depth 5 --seed 2
run cbq_s1  exp_cbq.py cbq_s1 --seed 1
run cbq_s2  exp_cbq.py cbq_s2 --seed 2
