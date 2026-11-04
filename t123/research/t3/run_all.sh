#!/bin/bash
# Reproduce the final T3 recipe from scratch (CPU, 4 threads; ~2.5 h). Run from repo root:  bash t123/research/t3/run_all.sh
set -e
R=t123/research/t3
# 1) classifier P(y>0): LightGBM, 3 seeds (seed 0 run also fits an un-offset LGB quantile model = E03)
python $R/exp_lgb_hurdle.py lgbh_log
python $R/exp_lgb_hurdle.py clf_s1 --clf_only --seed 1
python $R/exp_lgb_hurdle.py clf_s2 --clf_only --seed 2
# 2) positive-part quantile models on r = log y - log prev_month (offset), 11 alphas
for s in 0 1 2; do
  python $R/exp_cbq.py cbq3_s$s --seed $s --lr 0.03                                  # CatBoost MultiQuantile d8
  python $R/exp_xgbq.py xq_d5s$s --offset prev --extra --depth 5 --seed $s          # XGBoost multi-quantile d5
done
# 3) equal-weight geometric mean of the quantile models, 3-seed p, hurdle median (no tuning -> no nesting needed)
python $R/avg_q.py final --q cbq3_s0,cbq3_s1,cbq3_s2,xq_d5s0,xq_d5s1,xq_d5s2 --p lgbh_log,clf_s1,clf_s2
python $R/write_best.py final
