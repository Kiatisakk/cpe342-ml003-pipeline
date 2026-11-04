#!/bin/bash
cd "$(dirname "$0")/../../.."
L=t123/research/t3/logs
run(){ n=$1; shift; python -u t123/research/t3/$@ 2>&1 | grep -E --line-buffered "fold|RESULT|Error|Traceback|AUC" > $L/$n.log; }
run cbq_d4    exp_cbq.py cbq_d4 --depth 4 --lr 0.05
run cbq_d6s1  exp_cbq.py cbq_d6s1 --depth 6 --lr 0.04 --seed 1
run cbq_d6s2  exp_cbq.py cbq_d6s2 --depth 6 --lr 0.04 --seed 2
