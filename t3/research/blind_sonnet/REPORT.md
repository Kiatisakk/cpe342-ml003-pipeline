# T3 blind re-derivation (KFold 5, shuffle, rs=42; score = 1/(1+MAE/mean|y|))
EDA: 48% zeros; 15 rows at the 300000 cap. prev_month_spending is never missing/zero and, for y>0, y/prev ~ segment-dependent (seg0 ~1.2, seg1 ~2, seg2 ~2.5, seg3 ~5; whales heavy tail). Exact identities: historical_spending = total_transactions*avg_transaction_value (used to fill NaN, ~5.8% missing per column), purchases_on_discount = total_transactions*discount_rate_used. Other columns ~5.8% MCAR NaN. Zero probability falls with segment (72% -> 7%).
MAE-optimal prediction = conditional median: 0 if P(y>0)<0.5 else the (1-1/(2p)) quantile of the positive part.

| experiment | mean | std | runtime s | verdict |
|---|---|---|---|---|
| lgb_l1 (LightGBM L1 on y, engineered feats) | 0.7856 | 0.0031 | 151 | best single, reference |
| hurdle_lgb_quantile (clf + 11 quantile regs of y/prev, quantile rule) | 0.7838 | 0.0038 | 832 | good, diverse |
| nn_mlp_l1 (MLP 256-128-64, L1 loss, torch) | 0.7762 | 0.0026 | n/m | decent, adds to blend |
| knn_wmedian (k=40 weighted median of y/prev) | 0.7578 | 0.0055 | 147 | weak |
| extratrees_ratio (weighted mean of y/prev) | 0.7578 | 0.0061 | 1083 | weak (mean not median) |
| lgb_ratio_wl1 (L1 on y/prev, weight prev) | 0.7688 | 0.0056 | n/m | worse than direct L1 |
| svr_nys_0.02_600 (Nystroem 600 + SGD eps-insens eps=0 on ratio, weighted) | 0.6388 | 0.0429 | 179 | poor, unstable |
| svr_linear (linear eps-insens SGD on ratio) | 0.5756 | 0.0748 | 17 | poor (hurdle structure not linear) |
| stack_lad (LAD weights over all, weights fit in-fold on OOF) | 0.7878 | 0.0032 | - | best; +0.002 over lgb_l1 |
| mean(lgb_l1, hurdle) | 0.7869 | - | - | |
| mean(lgb_l1, hurdle, nn) | 0.7870 | - | - | |

Top 3: stack_lad, lgb_l1, hurdle_lgb_quantile. Test preds = mean of 5 fold models (stack weights fit on all OOF). SVR was run with limited tuning (single gamma/alpha, no kernel SVR on full data); not exhaustively tuned. Timings for NN not recorded (CPU contention). Nothing submitted.
