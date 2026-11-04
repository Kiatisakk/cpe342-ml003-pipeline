#!/bin/bash
# sequential experiment queue (run from repo root)
cd "$(dirname "$0")/../../.."
L=t123/research/t3/logs
run(){ n=$1; shift; python -u t123/research/t3/$@ 2>&1 | grep -E --line-buffered "fold|RESULT|Error|Traceback" > $L/$n.log; }
run xq_extra  exp_xgbq.py xq_extra --offset prev --extra
run xq_lin    exp_xgbq.py xq_lin --offset lin --extra
run xq_w05    exp_xgbq.py xq_w05 --offset prev --extra --wscale 0.5
run xq_linfull exp_xgbq.py xq_linfull --offset linfull --extra
run cbq_prev exp_cbq.py cbq_prev
