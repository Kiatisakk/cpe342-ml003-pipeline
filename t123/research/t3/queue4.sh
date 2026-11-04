#!/bin/bash
cd "$(dirname "$0")/../../.."
L=t123/research/t3/logs
run(){ n=$1; shift; python -u t123/research/t3/$@ 2>&1 | grep -E --line-buffered "fold|RESULT|Error|Traceback|AUC" > $L/$n.log; }
run cbq3_s1  exp_cbq.py cbq3_s1 --seed 1 --lr 0.03
run cbq3_s2  exp_cbq.py cbq3_s2 --seed 2 --lr 0.03
t123/research/t3/queue5.sh
