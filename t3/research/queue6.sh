#!/bin/bash
cd "$(dirname "$0")/../../.."
L=t3/research/logs
run(){ n=$1; shift; python -u t3/research/$@ 2>&1 | grep -E --line-buffered "fold|RESULT|Error|Traceback|AUC" > $L/$n.log; }
run cbq_d6  exp_cbq.py cbq_d6 --depth 6 --lr 0.04
run cbq_d10 exp_cbq.py cbq_d10 --depth 10 --lr 0.03
