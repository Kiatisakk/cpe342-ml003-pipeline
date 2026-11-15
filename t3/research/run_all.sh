#!/bin/bash
# Reproduce the FINAL T3 recipe from scratch (CPU, 4 threads, ~1 h). Run from repo root:  bash t3/research/run_all.sh
set -e
R=t3/research
# 1) classifier P(y>0): LightGBM binary, 3 seeds (the seed-0 run also fits the plain LGB quantile model of E03)
python $R/exp_lgb_hurdle.py lgbh_log
python $R/exp_lgb_hurdle.py clf_s1 --clf_only --seed 1
python $R/exp_lgb_hurdle.py clf_s2 --clf_only --seed 2
# 2) positive part: CatBoost MultiQuantile (11 alphas), depth 4, lr .05, on r = log y - log prev_month_spending
python $R/exp_cbq.py cbq_d4   --depth 4 --lr 0.05
python $R/exp_cbq.py cbq_d4s1 --depth 4 --lr 0.05 --seed 1
python $R/exp_cbq.py cbq_d4s2 --depth 4 --lr 0.05 --seed 2
# 3) equal-weight geometric mean of the 3 quantile models, mean of 3 p's, hurdle median (no tuned weights -> no nesting needed)
python $R/avg_q.py cbd4_bag3 --q cbq_d4,cbq_d4s1,cbq_d4s2 --p lgbh_log,clf_s1,clf_s2
python $R/write_best.py cbd4_bag3
