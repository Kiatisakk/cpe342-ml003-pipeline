#!/bin/bash
# learning curves: amount model (CB d4) and classifier (LGB) on a fraction of each training fold
cd "$(dirname "$0")/../../.."
L=t3/research/logs
run(){ n=$1; shift; python -u t3/research/$@ 2>&1 | grep -E --line-buffered "RESULT|Error|Traceback|AUC" > $L/$n.log; }
for f in 0.25 0.5 0.75; do run lc_cb_$f exp_cbq.py lc_cb_$f --depth 4 --lr 0.05 --p_avg --frac $f; done
for f in 0.25 0.5 0.75; do run lc_clf_$f exp_lgb_hurdle.py lc_clf_$f --clf_only --frac $f; done
run clf_cb6  exp_clf.py clf_cb6 --algo cb --depth 6
run clf_cb4  exp_clf.py clf_cb4 --algo cb --depth 4
run clf_xgb5 exp_clf.py clf_xgb5 --algo xgb --depth 5 --lr 0.03
