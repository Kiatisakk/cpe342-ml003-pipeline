#!/bin/bash
cd "$(dirname "$0")/../../.."
L=t123/research/t3/logs
run(){ n=$1; shift; python -u t123/research/t3/$@ 2>&1 | grep -E --line-buffered "fold|RESULT|Error|Traceback" > $L/$n.log; }
run xq_w3     exp_xgbq.py xq_w3 --offset prev --extra --subset seg3
python t123/research/t3/mix_whale.py xq_extra xq_w3 mix_w3 > $L/mix_w3.log 2>&1
run xq_lr02   exp_xgbq.py xq_lr02 --offset prev --extra --lr 0.02
run xq_d5     exp_xgbq.py xq_d5 --offset prev --extra --depth 5
run xq_d9     exp_xgbq.py xq_d9 --offset prev --extra --depth 9 --mcw 10
run direct    exp_direct.py direct
run cbq_lr03  exp_cbq.py cbq_lr03 --lr 0.03
t123/research/t3/queue3.sh
