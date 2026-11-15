#!/bin/bash
cd "$(dirname "$0")/../../.."
L=t3/research/logs
run(){ n=$1; shift; python -u t3/research/$@ 2>&1 | grep -E --line-buffered "fold|RESULT|Error|Traceback" > $L/$n.log; }
run xq_w3     exp_xgbq.py xq_w3 --offset prev --extra --subset seg3
python t3/research/mix_whale.py xq_extra xq_w3 mix_w3 > $L/mix_w3.log 2>&1
run xq_lr02   exp_xgbq.py xq_lr02 --offset prev --extra --lr 0.02
run xq_d5     exp_xgbq.py xq_d5 --offset prev --extra --depth 5
run xq_d9     exp_xgbq.py xq_d9 --offset prev --extra --depth 9 --mcw 10
run direct    exp_direct.py direct
run cbq_lr03  exp_cbq.py cbq_lr03 --lr 0.03
t3/research/queue3.sh
