# Task 3 (spending_30d) — research log, branch `research/t3`

Metric: NMAE score = 1 / (1 + Σ|y − ŷ| / Σy) (higher is better). All numbers are out-of-fold (OOF) on the **same 5 folds as v4**
(`StratifiedKFold(5, shuffle, seed 42)` on zero / payer-quartile strata, reproduced in `common.folds_v4`). "fold mean ± sd" is over the 5 folds.

## Result

| | OOF | fold mean ± sd | paired gain vs honest baseline (per fold) |
|---|---|---|---|
| v4 D_q alone (repo OOF) | 0.79169 | 0.79174 ± 0.0075 | |
| v4 blend, greedy weights tuned in-sample (the reported 0.7920) | 0.79195 | 0.79201 ± 0.0077 | optimistic |
| **honest baseline**: v4 A, C, D_q, D_qw0.3, D_qw0.6 with nested LAD blend weights | **0.79191** | 0.79196 ± 0.0077 | — |
| **final: CatBoost depth-4 offset multi-quantile hurdle, 3 seeds** | **0.79529** | **0.79534 ± 0.0076** | **+0.00337 ± 0.00043** (+0.0038, +0.0033, +0.0038, +0.0026, +0.0034) |

The final recipe has **no tuned weights, thresholds or calibration** (equal-weight bag; the hurdle rule q = 1 − 1/(2p) is fixed),
so its OOF score needs no nesting. The only selection made on OOF was the hyper-parameter choice (offset type, depth, lr) among the
variants listed below, and the pick among bag compositions (E35–E38 within 0.00012 of each other), so the selection bias is
≈ 0.0001–0.0002, far below the +0.0034 gain. (The old 0.7925 v4+NN blend also used in-sample weights; the new model beats it as well.)
Error ratio Σ|e|/Σy per segment, v4 → new: seg 0 0.989 → 0.992, seg 1 0.731 → 0.727, seg 2 0.487 → 0.482, **seg 3 (whales) 0.2385 → 0.2330**.

Files: `best_oof.csv` (y, pred; train row order) and `t3_best_test.csv` (id ANS00001..ANS25889 in test.csv order, pred float ≥ 0; 50.4 % zeros, mean 10,430).

## Final recipe (`bash t123/research/t3/run_all.sh`, CPU ~1 h)
1. Preprocessing = v4 exactly (`common.py`): exact identities (historical_spending = total_transactions × avg_transaction_value,
   purchases_on_discount = total_transactions × discount_rate_used, and their inverses), v4 ratio/log features, segment imputed
   by an XGB classifier fitted on train rows with known segment (no target, no test rows), p_whale feature. + 4 log-ratio features (`extra_feats`).
2. P(y>0): LightGBM binary (lr 0.03, 63 leaves, early stopping on a 10 % split of the training fold), mean of 3 seeds. AUC 0.784.
3. Positive part: **CatBoost MultiQuantile** (alphas 0.02…0.5, 11 levels), depth 4, lr 0.05, l2 3, early stopping (200),
   trained on payers of the training fold with **offset target r = log y − log prev_month_spending**
   (fallback log historical_spending − 1.05, then 6.5, when prev is missing). Quantiles of y = exp(Q_r + offset).
   3 seeds, geometric mean of the quantiles.
4. Prediction = mixture median: 0 if p ≤ 0.5, else the positive-part quantile at q = 1 − 1/(2p) (linear interpolation over alphas).
   Test = average of the 5 fold models.

## What mattered (and what did not)
* **Offset target log(y / prev_month) is the main win** (+0.0023 XGB, +0.0039 LGB). log y ≈ log prev + segment multiplier
  (corr 0.88–0.96 inside segments); with the offset the trees only model the multiplier instead of re-building the identity line from splits.
  A fitted linear offset (E08, E10) or log-historical offset (E25) was not better than the unit-coefficient prev offset.
* **Shallow trees** for the residual: CatBoost d4 ≈ d3 ≈ d6 > d8; XGB d5 ≥ d7 > d9, d3 worse. CatBoost > XGB > LGB.
* Seed bagging: +0.0002–0.0003. Mixing model families or LAD stacking on top gave nothing once the base model was good (E11, E28, E37).
* Zeros are essentially unpredictable *within* a segment (AUC 0.55–0.63 per segment, 0.784 overall); zero-spend whales are 27 % of
  the whale error and are irreducible, so classifier work (seed bagging) gave nothing.
* No gain: per-segment (shift, κ) calibration (already calibrated), feature-only scale weights, whale-specialist model,
  direct log-link MAE on raw y, lower learning rate, deeper trees.
* Not run: neural nets (PyTorch could not be installed — download.pytorch.org blocked by the sandbox proxy); the NN-ZILN OOF is not in the repo.

## Rule compliance
* Train from scratch only (XGBoost / LightGBM / CatBoost / sklearn / scipy); no pretrained models, no AutoML; no tuning library needed.
* test.csv is only transformed (exact fill, deterministic features, `predict`); no imputer/encoder/model is fitted on test rows; no pseudo-labels.
* The segment imputer is fitted once on all train rows with known segment (features only, no target), as in v4.
* Early stopping uses an inner 10 % split of each training fold, never the validation fold. Nothing submitted to Kaggle.

## Every experiment
| id | experiment | OOF NMAE | fold mean ± sd | decision | note |
|---|---|---|---|---|---|
| E00 | v4 D_q alone (repo OOF) | 0.79169 | 0.79174 +- 0.0075 | ref | baseline single |
| E01 | v4 blend, in-sample greedy weights | 0.79195 | 0.79201 +- 0.0077 | ref | weights tuned on same OOF (optimistic) |
| E02 | v4 A,C,D_q,D_qw0.3,D_qw0.6 nested LAD blend | 0.79191 | 0.79196 +- 0.0077 | ref | HONEST baseline (nested) |
| E03 | LightGBM hurdle: binary clf + 11 per-alpha quantile regs on log y (lr .03, 63 leaves) | 0.79002 | 0.79007 +- 0.0076 | discard | clf AUC 0.7838 (= v4 0.7840); amount model weaker than v4 XGB multi-quantile |
| E04 | nested per-segment (p-shift, kappa) calibration on E03 | 0.79002 | 0.79007 +- 0.0076 | discard | all segments pick shift 0 / kappa 1 -> already calibrated |
| E05 | XGB multi-quantile log y (lr .05 d7), LGB clf p | 0.79162 | 0.79167 +- 0.0077 | ref | reproduces v4 D_q (0.79169) in this harness |
| E06 | E05 + offset target log y - log prev_month (fallback log hist - 1.05) | 0.79391 | 0.79396 +- 0.0076 | KEEP | +0.0023 vs E05; trees model residual multiplier, fewer iters |
| E07 | E06 + 4 log-ratio feats (lprev-lhist, l_atv*freq, l_hist/tx, l_hist/age) | 0.79405 | 0.79410 +- 0.0077 | neutral | +0.00014 = noise; kept as default feature set |
| E08 | offset = fold-fitted linear model (log prev/hist/atv x segment) instead of log prev | 0.79223 | 0.79228 +- 0.0075 | discard | -0.0018 vs E07: unit-coef log prev offset is better than fitted mean offset |
| E09 | E07 + feature-only weight exp(0.5*log prev) | 0.79263 | 0.79268 +- 0.0078 | discard | -0.0014: up-weighting large rows hurts |
| E10 | offset = per-segment ridge on all feats (fold-fitted) | 0.79283 | 0.79289 +- 0.0079 | discard | -0.0012 vs E07; kept only as stacking candidate |
| E11 | nested LAD stack of 12 bases (E05-E10, LGB hurdle, v4 cols) | 0.79421 | 0.79427 +- 0.0078 | neutral | +0.00016 over E07 = noise; weights mostly xq_extra/xq_prev/xq_linfull |
| E12 | CatBoost MultiQuantile (11 alphas) d8 lr .06, offset log prev + extra feats | 0.79434 | 0.79439 +- 0.0079 | KEEP (best single) | +0.0003 vs E07 (borderline); stops at ~300 iters |
| E13 | whale specialist (XGB-q on seg-3 payers, offset) geometric-mixed with E07 by p_whale | 0.79148 | 0.79154 +- 0.0081 | discard | -0.0026: specialist has less data, general model already shares structure |
| E14 | E07 with lr 0.02 | 0.79402 | 0.79407 +- 0.0075 | discard | same as lr .05, 2.5x slower |
| E15 | E07 with depth 5 | 0.79424 | 0.79429 +- 0.0076 | neutral | +0.0002 vs d7 (noise); slightly shallower is fine |
| E16 | E07 with depth 9, mcw 10 | 0.79321 | 0.79325 +- 0.0076 | discard | deeper is worse; d5 >= d7 > d9 |
| E17 | direct log-link MAE on raw y, all rows, offset log prev (custom XGB obj) | 0.77949 | 0.77953 +- 0.0071 | discard | sign-gradient ES stops at ~55 trees; cannot reach exact zeros (frac<1 = 0.1%) |
| E18 | CatBoost-q offset, lr .03 | 0.79449 | 0.79454 +- 0.0076 | KEEP (best single) | +0.00015 vs lr .06 (noise) |
| E19 | E15 with classifier p averaged over 3 LGB seeds | 0.79429 | 0.79434 +- 0.0076 | neutral | +0.00005 (noise); 3-seed p used from here (harmless) |
| E20 | XGB-q d5 offset, 3 seeds bagged (geo-mean quantiles) + 3-seed p | 0.79446 | 0.79451 +- 0.0076 | KEEP | +0.0002 over single seeds (0.79416-0.79431); variance reduction |
| E21 | CatBoost-q offset bag of 4 (seeds 0,1,2 lr .06 + lr .03) | 0.79472 | 0.79476 +- 0.0076 | KEEP | +0.0002 over best single CB |
| E22 | equal-weight geo-mean of 7 quantile models (4 CB + 3 XGB d5) + 3-seed p | 0.79490 | 0.79495 +- 0.0076 | KEEP (BEST) | no tuning -> no nesting needed |
| E23 | CatBoost-q lr .03 bag of 3 seeds | 0.79476 | 0.79480 +- 0.0076 | part of final | single seeds 0.79449/0.79462/0.79450 |
| E24 | FINAL: geo-mean of 6 quantile models (3 CB lr .03 + 3 XGB d5) + 3-seed p | 0.79492 | 0.79496 +- 0.0076 | KEEP (BEST) | = run_all.sh recipe; adding 3 CB lr .06 -> 0.79493 (saturated) |
| E25 | XGB-q d5, offset log hist - 1.05 (instead of log prev) | 0.79421 | 0.79426 +- 0.0076 | discard | = prev offset alone; adding to final bag -> 0.79490 (no gain) |
| E26 | LightGBM per-alpha quantile, offset log prev, 31 leaves lr .05, 3-seed p | 0.79390 | 0.79395 +- 0.0077 | discard (as member) | offset +0.0039 vs E03; adding to final bag -> 0.79489 (no gain) |
| E27 | arithmetic mean of the 6 models' hurdle predictions (instead of geo-mean of quantiles) | 0.79486 | - | discard | equivalent to E24 (-0.00006) |
| E28 | nested LAD over family bags (CB3, XQ3, LGB-offset, linfull) | 0.79485 | 0.79490 +- 0.0077 | discard | = E24; equal weights are as good and need no tuning |
| E29 | CatBoost-q offset depth 6 lr .04 (single seed) | 0.79497 | 0.79502 +- 0.0076 | KEEP | +0.0004 vs d8 singles; shallower better again (d10 run aborted) |
| E30 | CatBoost-q offset depth 4 lr .05 (single seed) | 0.79505 | 0.79510 +- 0.0076 | KEEP | best single; d4 >= d6 > d8 |
| E31 | CatBoost-q d6 bag of 3 seeds | 0.79508 | 0.79513 +- 0.0076 | KEEP | singles 0.79482-0.79497 |
| E32 | CatBoost-q offset depth 3 lr .06 | 0.79507 | 0.79512 +- 0.0075 | neutral | = d4/d6; depth curve flat 3-6, worse at 8 |
| E33 | CatBoost-q offset depth 4 seeds 1,2 | 0.79515 / 0.79518 | - | part of final | seed 0 = 0.79505 |
| E34 | XGB-q offset depth 3 | 0.79410 | 0.79415 +- 0.0076 | discard | XGB best at d5 |
| E35 | CatBoost d4 bag of 3 seeds + 3-seed p  (FINAL) | 0.79529 | 0.79534 +- 0.0076 | KEEP (BEST) | paired vs nested v4: +0.00337 +- 0.00043, all 5 folds positive |
| E36 | CB d3/d4/d6 bag of 7 | 0.79523 | 0.79528 +- 0.0076 | neutral | = E35 |
| E37 | CB shallow 7 + XGB d5 x3 (10 models) | 0.79524 | 0.79529 +- 0.0076 | neutral | = E35 (+0.00333 paired); XGB adds no diversity once CB is shallow |
| E38 | CB d4 x3 + XGB d5 x3 | 0.79517 | 0.79522 +- 0.0076 | neutral | = E35 |

## Scripts
`common.py` (data, folds, metric, hurdle median), `exp_lgb_hurdle.py` (LGB clf + per-alpha quantiles), `exp_xgbq.py` (XGB multi-quantile, offsets,
weights, whale subset, seed bag), `exp_cbq.py` (CatBoost MultiQuantile), `exp_direct.py` (direct log-link MAE), `calib_nested.py`, `mix_whale.py`,
`avg_q.py` (bagging + hurdle), `stack_lad.py` (nested LAD blend via LP), `write_best.py`, `queue*.sh` (the exact run order used), `run_all.sh`.
OOF/test arrays are written to `oof/` (git-ignored, regenerate with the scripts); data cache in `cache/` (git-ignored).
