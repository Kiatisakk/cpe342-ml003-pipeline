#!/bin/bash
cd "$(dirname "$0")/../../.."
L=t3/research/logs
run(){ n=$1; shift; python -u t3/research/$@ 2>&1 | grep -E --line-buffered "fold|RESULT|Error|Traceback|AUC" > $L/$n.log; }
C="--depth 4 --lr 0.05 --p_avg"
run cb_fine  exp_cbq.py cb_fine  $C --alphas 0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.34,0.38,0.41,0.43,0.45,0.46,0.47,0.48,0.49,0.5
run cb_ratio exp_cbq.py cb_ratio $C --ratio
run cb_l2_10 exp_cbq.py cb_l2_10 $C --l2 10
run cb_lr03  exp_cbq.py cb_lr03  --depth 4 --lr 0.03 --p_avg
run nnq exp_nnq.py nnq --seeds 2 --epochs 60
t3/research/queue9.sh
