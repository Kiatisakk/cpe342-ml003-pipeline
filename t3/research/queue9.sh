#!/bin/bash
# robustness: different fold split (seed 7) - v4-equivalent baseline vs final recipe, same folds
cd "$(dirname "$0")/../../.."
L=t3/research/logs
export FOLD_SEED=7
run(){ n=$1; shift; python -u t3/research/$@ 2>&1 | grep -E --line-buffered "fold|RESULT|Error|Traceback|AUC" > $L/$n.log; }
run f7_clf0 exp_lgb_hurdle.py f7_clf0 --clf_only --seed 0
run f7_xq_none exp_xgbq.py f7_xq_none --offset none --p f7_clf0
run f7_cb_d4 exp_cbq.py f7_cb_d4 --depth 4 --lr 0.05 --p f7_clf0
