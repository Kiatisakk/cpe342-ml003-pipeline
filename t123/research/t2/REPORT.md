# T2 (player segment, 4 classes, macro-F1) — research report

Branch `research/t2`. Every script runs from the repo root (`python t123/research/t2/<script>.py`).
Validation: 5-fold StratifiedKFold, seeds 42 / 7 / 123 (same as `t123/t2_nb/pf.py`); mean ± sd over seeds.
Anything tuned on labels (class bias, meta-learners, WANBIA weights, pooling choice) is nested inside the outer folds.
Note: the seed-to-seed sd (~0.0004) is computed on the same 104k rows. The real sampling noise of macro-F1
on a dataset this size is about ±0.002, so differences below ~0.001 count as ties.

## TL;DR

| | macro-F1 (OOF, 3 seeds) |
|---|---|
| Baseline NB, per-feature Optuna cfg (reproduced exactly: 0.7562 / 0.7555 / 0.7553) | **0.7557 ± 0.0004** |
| Best feature-only model: label-noise-aware EM-NB (with nested bias: 0.7557 ± 0.0002) | **0.7557 ± 0.0004** (tie), log-loss 0.602 vs 0.624 |
| **player_id block rule (ID LEAK, see §2)** | **0.9677 ± 0.0000** |

**Feature-only: no technique beats the baseline.** Every feature-only model lands within 0.0005 of 0.7557.
The data explains why: features are independent given the class (|within-class Spearman| < 0.05),
so NB with good densities is Bayes-optimal, and the score is the Bayes ceiling (≈ 0.756 on noisy labels).

**The only real gain is a leak:** `player_id` was assigned in segment blocks. It gives 0.9677, which is
the ceiling because 3% of labels are random flips. Whether to use it is a team / course-rules decision (§2).

## 1. Experiments

| # | script | experiment | macro-F1 (mean ± sd) | log-loss | kept? | note |
|---|---|---|---|---|---|---|
| E00 | e00_baseline.py | NB, uniform cfg (32 bins, σ 0.7) | 0.7550 ± 0.0004 | — | ref | matches nb.py exactly (0.7548 on seed 42) |
| E00 | e00_baseline.py | NB, per-feature Optuna cfg (baseline) | 0.7557 ± 0.0004 | 0.6243 | **baseline** | cfg was tuned on seed-42 folds (slightly optimistic) |
| EDA | — | missingness as a class signal | — | — | discarded | NaN rate 7.5–8% for every feature and class; NaN count per row is equal across classes, so missingness is MCAR |
| EDA | — | do the 16 noise columns carry signal? | — | — | discarded | univariate NB info gain ≤ 0 for all 16 |
| E01 | e01_bias.py | nested macro-F1 class-bias on NB log-posterior | 0.7556 ± 0.0001 | — | discarded | tuned biases ≈ 0, so argmax is already F1-optimal |
| E02 | e02_gam.py | multinomial spline-GAM (8 quantile knots, C=1) = discriminative NB | 0.7529 ± 0.0005 (nested bias 0.7532) | 0.6198 | discarded | better LL, worse F1; 12/20 knots and C 0.05 also worse on fold 0 |
| E04 | e04_wanbia.py w | WANBIA, one weight per feature (honest inner-OOF blocks) | 0.7541 (s42), nested bias 0.7555 | 0.6201 | discarded | — |
| E04 | e04_wanbia.py wk | WANBIA, per feature × class weights | 0.7542 (s42) | 0.6207 | discarded | — |
| E04 | e04_wanbia.py lr | multinomial LR on all centred NB log-lik blocks | 0.7539 (s42) | 0.6195 | discarded | — |
| E05 | e05_lgbm.py raw | LightGBM on raw features | RAW_RES | | discarded | probes for interactions (CI violations) |
| E05 | e05_lgbm.py nbf | LightGBM on raw + honest inner-OOF NB log-lik ratios | NBF_RES | | discarded | |
| E06 | e06_emnb.py | **label-noise-aware NB by EM** (latent true class, symmetric noise ε) | 0.7557 ± 0.0004 (nested bias 0.7557 ± 0.0002) | **0.6017** | kept for probabilities | ε estimated from the labels alone is 0.028 (true value 0.030, §2). Same argmax as NB, much better calibrated |
| E06 | e06_emnb.py idclean | diagnostic: NB trained on ID-cleaned labels | 0.7560 (s42) | — | diagnostic | clean labels do not change argmax decisions |
| E07 | e07_nbbag.py | multi-resolution NB (average log-densities of 6 bin/σ configs) | 0.7555 ± 0.0003 | 0.6230 | discarded | — |
| E08 | e08_blend.py | geometric mean NB + EM-NB + GAM | 0.7556 ± 0.0001 | 0.6106 | discarded | — |
| E08 | e08_blend.py | nested LR stack of NB + EM-NB + GAM | STACK_RES | | discarded | |
| E09 | e09_2x2.py | 2×2-factorised NB (pool each feature's density over an axis, chosen per fold) | 0.7550 (s42) | 0.6245 | discarded | the data rejects pooling for 26/28 features (small cross-axis effects exist) |
| E10 | e10_idblock.py | **player_id block rule + NB tie-break in boundary gaps** | **0.9677 ± 0.0000** | 0.168 | separate deliverable (LEAK) | see §2 |

## 2. player_id leak and label noise (important for the team)

* `player_id` = `P` + number in 0..129,999, shared between train (104,000) and test (25,889). There is no overlap, and the ids interleave.
* Ids were assigned in segment blocks sized 40 / 25 / 20 / 15% of 130,000:
  `0–51,999 → 0`, `52,000–84,499 → 1`, `84,500–110,499 → 2`, `110,500–129,999 → 3`. The cut points learned from train fall exactly on these values.
* 3.0% of train rows have a label that differs from their block. Those labels are spread **uniformly** over the 3 other classes.
  On those rows NB agrees with the *block* class 78.6% of the time (its normal accuracy) and with the label only 7%.
  So they are **random label flips**: the features follow the block class. Expect the same 3% noise in the test labels.
* Consequence: id block → macro-F1 0.968, which is the ceiling (flips are unpredictable).
  The feature-only ceiling is ≈ 0.756. Only 1 test id falls inside a cut-point gap (decided by NB).
* T1 and T3 ids show no such pattern (Spearman of id vs target ≈ 0).
* **Rules:** using the id does not use test.csv for training and is not a pretrained model or AutoML.
  It is still a data-generation leak, not a model. The course page (p.15) does not mention it, so ask the TA or
  decide as a team before submitting. If used, say so in the report.
  The LB gain would be ≈ w_T2 × (0.968 − 0.756) ≈ +0.05 at w ≈ 0.25.

## 3. Final recipes / files

**Feature-only (rule-safe):** `best_oof_proba.npy` (seed-42 OOF, N×4), `best_test_proba.npy`, `t2_best_test.csv` (id,pred).
* Features: the baseline's (16 noise columns dropped). Per-feature quantile-histogram densities with the baseline's Optuna (bins, σ) config, Laplace α = 1, NaN marginalised.
* Fit: EM with a latent true class and symmetric label noise (15 iterations, ε → 0.028), on all train rows.
* Predict: P(label | x) = softmax(log π + Σ log p_j(x_j | z)) · M_ε, then argmax.
* CV 0.7557 ± 0.0004, the same as the baseline. Test predictions agree 97.6% with `preds_final/t2_nb.csv`.
  There is no reason to resubmit for F1; its probabilities are better calibrated (LL 0.602) if T2 is stacked anywhere.

**ID-leak variant (team decision):** `idleak_oof_proba.npy`, `idleak_test_proba.npy`, `t2_idleak_test.csv`.
CV 0.9677 ± 0.0000. Reproduce everything with `python t123/research/t2/make_final.py`.
Experiments cache to `t123/research/t2/cache/` (git-ignored).
