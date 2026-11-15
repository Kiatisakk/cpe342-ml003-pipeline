#!/bin/bash
# full 5-fold OOF of the 2 best Optuna configs (trial 20, trial 29), seed 0, 3-seed p; then in-sample (train-fit) scores
cd "$(dirname "$0")/../../.."
L=t3/research/logs
run(){ n=$1; shift; python -u t3/research/$@ 2>&1 | grep -E --line-buffered "fold|RESULT|Error|Traceback" > $L/$n.log; }
run opt_t20 exp_cbq.py opt_t20 --p_avg --depth 3 --lr 0.05 --l2 5.683 --rs 0.0514 --bt 0.273 --rsm 0.745 --border 128
run opt_t29 exp_cbq.py opt_t29 --p_avg --depth 3 --lr 0.05 --l2 8.748 --rs 0.217 --bt 0.330 --rsm 0.910 --border 128
run insample insample.py
