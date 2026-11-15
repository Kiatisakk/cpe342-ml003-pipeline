#!/bin/bash
cd "$(dirname "$0")/../../.."
L=t3/research/logs
run(){ n=$1; shift; python -u t3/research/$@ 2>&1 | grep -E --line-buffered "fold|RESULT|Error|Traceback|AUC" > $L/$n.log; }
run cbq3_s1  exp_cbq.py cbq3_s1 --seed 1 --lr 0.03
run cbq3_s2  exp_cbq.py cbq3_s2 --seed 2 --lr 0.03
t3/research/queue5.sh
