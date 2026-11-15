#!/bin/bash
cd "$(dirname "$0")/../../.."
L=t3/research/logs
run(){ n=$1; shift; python -u t3/research/$@ 2>&1 | grep -E --line-buffered "fold|RESULT|Error|Traceback|AUC" > $L/$n.log; }
run xq_hist  exp_xgbq.py xq_hist --offset hist --extra --depth 5
run lgbq_off exp_lgb_hurdle.py lgbq_off --offset prev --skip_clf --lr 0.05 --leaves 31
