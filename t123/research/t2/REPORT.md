# T2 (player segment, 4 classes, macro-F1) — research report

Branch `research/t2`. Every script runs from the repo root (`python t123/research/t2/<script>.py`).
Validation: 5-fold StratifiedKFold, seeds 42 / 7 / 123 (same as `t123/t2_nb/pf.py`); mean ± sd over seeds.
Anything tuned on labels (class bias, meta-learners, WANBIA weights, pooling choice) is nested inside the outer folds.
Note: the seed-to-seed sd (~0.0004) is computed on the same 104k rows. The real sampling noise of macro-F1
on a dataset this size is about ±0.002, so differences below ~0.001 count as ties.

## TL;DR

| | macro-F1 (OOF, 3 seeds) |
|---|---|
| Baseline NB, per-feature Optuna cfg (reproduced exactly: 0.7562 / 0.7555 / 0.7553) | 0.7557 ± 0.0004, **optimistic** |
| Same baseline, **nested** Optuna (honest, E13, seed 42) | 0.7544 |
| Baseline NB, untuned uniform cfg (32 bins, σ 0.7) | 0.7550 ± 0.0004 |
| **Best feature-only model (E14 g2 + EM): per-feature densities selected by inner-CV posterior log-loss + label-noise-aware EM-NB** | **0.7562 ± 0.0002** (0.7560 / 0.7564 / 0.7561), fully nested, log-loss 0.599 |
| player_id block rule (ID leak, see §2) | 0.9677 ± 0.0000 |

**Feature-only:** the honest gain is **+0.0012 over the untuned baseline and +0.0018 over nested Optuna**.
It is also higher than the over-optimistic 0.7557.
* The baseline's per-feature Optuna config overfits. All three seeds reuse the same 104k rows, so tuning 46
  hyperparameters on macro-F1 learns those rows: nested Optuna gives 0.7544, below the untuned 0.7550 (E13).
* `pf.py`'s "ll_select" has a bug. It compares histogram log-*mass* across different bin counts, which always
  favours coarse bins. Selecting each feature's (bins, σ) by inner-CV **univariate posterior log-loss** is the
  correct, low-variance criterion. It picks much finer and smoother densities (64–256 bins, σ 3–12)
  and needs no F1 tuning (E14).
* Modelling the 3% symmetric label noise with EM adds about +0.0004 and gives the best calibration (E06/E14).
* Everything else ties or loses: bias, GAM (±noise correction), WANBIA, GBMs, 2×2 pooling, stacking.
  Features are independent given the class, so NB is the right model. The learning curve (E11) shows only
  ≈ 0.002 headroom from more data, so ≈ 0.757 is close to the feature-only Bayes ceiling.

**The leak:** `player_id` was assigned in segment blocks. It gives 0.9677, which is the ceiling because
3% of labels are random flips that cannot be predicted (§2). Whether to use it is a team / course-rules decision.

## 1. Experiments

| # | script | experiment | macro-F1 (mean ± sd) | log-loss | kept? | note |
|---|---|---|---|---|---|---|
| E00 | e00_baseline.py | NB, uniform cfg (32 bins, σ 0.7) | 0.7550 ± 0.0004 | — | ref | matches nb.py exactly (0.7548 on seed 42) |
| E00 | e00_baseline.py | NB, per-feature Optuna cfg (baseline) | 0.7557 ± 0.0004 | 0.6243 | **baseline** | cfg was F1-tuned on these same rows; see E13 |
| EDA | — | missingness as a class signal | — | — | discarded | NaN rate 7.5–8% for every feature and class; NaN count per row is equal across classes, so missingness is MCAR |
| EDA | — | do the 16 noise columns carry signal? | — | — | discarded | univariate NB info gain ≤ 0 for all 16 |
| E01 | e01_bias.py | nested macro-F1 class-bias on NB log-posterior | 0.7556 ± 0.0001 | — | discarded | tuned biases ≈ 0, so argmax is already F1-optimal |
| E02 | e02_gam.py | multinomial spline-GAM (8 quantile knots, C=1) = discriminative NB | 0.7529 ± 0.0005 (nested bias 0.7532) | 0.6198 | discarded | better LL, worse F1; 12/20 knots and C 0.05 also worse on fold 0 |
| E04 | e04_wanbia.py w | WANBIA, one weight per feature (honest inner-OOF blocks) | 0.7541 (s42), nested bias 0.7555 | 0.6201 | discarded | — |
| E04 | e04_wanbia.py wk | WANBIA, per feature × class weights | 0.7542 (s42) | 0.6207 | discarded | — |
| E04 | e04_wanbia.py lr | multinomial LR on all centred NB log-lik blocks | 0.7539 (s42) | 0.6195 | discarded | — |
| E05 | e05_lgbm.py raw | LightGBM on raw features | 0.7509 (s42) | 0.6173 | discarded | GBMs find no interaction signal NB misses; their LL is better but F1 is worse |
| E05 | e05_lgbm.py nbf | LightGBM on raw + honest inner-OOF NB log-lik ratios | 0.7506 (s42) | 0.6161 | discarded | NB features do not help the GBM beat NB |
| E06 | e06_emnb.py | **label-noise-aware NB by EM** (latent true class, symmetric noise ε) | 0.7557 ± 0.0004 (nested bias 0.7557 ± 0.0002) | **0.6017** | kept for probabilities | ε estimated from the labels alone is 0.028 (true value 0.030, §2). Same argmax as NB, much better calibrated |
| E06 | e06_emnb.py idclean | diagnostic: NB trained on ID-cleaned labels | 0.7560 (s42) | — | diagnostic | clean labels do not change argmax decisions |
| E07 | e07_nbbag.py | multi-resolution NB (average log-densities of 6 bin/σ configs) | 0.7555 ± 0.0003 | 0.6230 | discarded | — |
| E08 | e08_blend.py | geometric mean NB + EM-NB + GAM | 0.7556 ± 0.0001 | 0.6106 | discarded | — |
| E08 | e08_blend.py | nested LR stack of NB + EM-NB + GAM | raw 0.7556 ± 0.0001, nested bias 0.7560 ± 0.0004 | 0.6005 | discarded | +0.0003 is below the 0.001 noise bar; the extra complexity is not worth it |
| E09 | e09_2x2.py | 2×2-factorised NB (pool each feature's density over an axis, chosen per fold) | 0.7550 (s42) | 0.6245 | discarded | the data rejects pooling for 26/28 features (small cross-axis effects exist) |
| E10 | e10_idblock.py | **player_id block rule + NB tie-break in boundary gaps** | **0.9677 ± 0.0000** | 0.168 | separate deliverable (LEAK) | see §2 |
| E11 | e11_learning_curve.py | learning curve, NB / EM-NB at 10–100% of fold-train | 0.746 → 0.752 → 0.754 → 0.755 → 0.7557 | — | diagnostic | +0.0018 per doubling and shrinking, so ≈ 0.002 headroom from unlimited data. EM helps most at small n |
| — | (inline) | T3 train has a `segment` column: use it as extra T2 data? | — | — | discarded | different generator (e.g. playtime mean ≈ 500 vs 1,500) and only 7 shared columns |
| E12 | e12_gam_nc.py | spline-GAM with forward label-noise correction (torch, ε learned = 0.027) | 0.7525 (s42) | 0.6029 | discarded | the noise correction fixes LL, not F1. GAM accuracy really is lower (0.768 vs 0.771) |
| — | (inline) | sharp edges / jumps in class densities? (100-bin class fractions, z-test of adjacent bins) | — | — | discarded | max z 3–4.7 over ~400 tests per feature, consistent with noise: densities are smooth |
| — | (inline) | parametric densities (truncated normal / gamma / lognormal per class) | — | — | discarded | truncnorm fits best (KS ≈ 0.01) but is still detectably wrong: lighter upper tail, extra mass at the lower edge |
| E13 | e13_nested_optuna.py | **nested** per-feature Optuna (150 TPE trials on inner 4-fold CV per outer fold, seed 42) | 0.7544 (s42); same folds: uniform 0.7548, original Optuna 0.7562 | — | diagnostic | the baseline's +0.0007 to +0.0014 tuning gain is overfitting |
| E14 | e14_llselect.py <s> g1 0 | per-feature (bins, σ) by inner-CV univariate posterior log-loss, grid bins 8–128 × σ 0–4 | 0.7554 ± 0.0001 | 0.6229 | superseded | picks sit at the 128-bin / σ 4 edge of the grid |
| E14 | e14_llselect.py <s> g2 0 | same, extended grid bins {64,128,256} × σ {3,4,6,8,12} | 0.7558 ± 0.0002 (nested bias 0.7561) | 0.6229 | superseded | — |
| E14 | e14_llselect.py <s> g2 1 | **g2 + label-noise EM-NB** | **0.7562 ± 0.0002** (nested bias 0.7560) | **0.5993** | **KEPT (final feature-only)** | honest +0.0012 vs uniform baseline, +0.0018 vs nested Optuna |
| E14 | e14_llselect.py 42 g3 1 | grid further out, bins {128,256,512} × σ {6..24} | 0.7557 (s42 vs 0.7560 for g2) | — | discarded | over-smoothing |
| E08 | e08_blend.py | geo-mean / nested LR stack of the E14 + EM-NB variants | 0.7561 / 0.7560 | 0.6075 / 0.5996 | discarded | no gain over E14 g2+EM alone |
| E15 | e15_specialists.py | class specialists: binary LightGBM "class k vs rest" (raw features + honest NB log-posteriors). Rules: hier0 (class-0 specialist first, then NB among 1–3), swap0, OvR argmax, OvR+NB | NB 0.7550 · hier0 0.7537 · swap0 0.7544 · OvR 0.7539 · OvR+NB 0.7551 (3 seeds, nested) | — | discarded | specialists' one-vs-rest AUC equals NB's (0.931 / 0.923 / 0.924 / 0.940): a dedicated class-0 model learns nothing NB lacks |
| E16 | e16_weight_blend.py | **nested weight blending** of 6 feature-only members (E14 g2±EM, EM-NB base, GAM, GAM-nc, LGBM). Weights fitted on the other 4 folds: linear pool by log-loss, log-linear pool by log-loss, log-linear by macro-F1, equal geo-mean (± nested bias) | best single 0.7562 ± 0.0002 · lin_ll 0.7563 ± 0.0001 · geo_ll 0.7562 · geo_f1 0.7558 · equal 0.7557 | 0.5991 vs 0.5992 | tie, optional file | log-loss weights are stable (≈ 0.88 E14 + 0.11 GAM-nc, sd 0.02–0.05). F1-fitted weights are unstable (sd 0.2) and worse. Fold sd unchanged (0.0023), worst fold 0.7498 vs 0.7493. Test agreement 99.7%. Saved as `t2_blend_test.csv` / `blend_test_proba.npy` |
| — | (inline) | can the 3% label flips be predicted? (every column, id mod m, id/row position, neighbour labels) | — | — | discarded | no: flip rate is flat everywhere, a flipped label matches neighbour labels only at chance (0.24), flips are uniform over other classes (χ² p 0.44–0.84) |

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

**Feature-only (rule-safe):** `best_oof_proba.npy` (seed-42 nested OOF, N×4), `best_test_proba.npy`, `t2_best_test.csv` (id,pred), `best_cfg.json`.
1. Features: the baseline's (16 noise columns dropped). NaN is marginalised for continuous features and is its own level for discrete ones.
2. Densities: per continuous feature, quantile-histogram (bins, Gaussian σ in bin units), chosen from bins {64,128,256} × σ {3,4,6,8,12}
   by 4-fold CV univariate posterior log-loss on the training rows (no F1 tuning). Laplace α = 1.
3. Fit: EM with a latent true class and symmetric label noise (15 iterations, ε → 0.0285 on all train).
4. Predict: P(label | x) = softmax(log π + Σ log p_j(x_j | z)) · M_ε, then argmax.
* CV 0.7562 ± 0.0002, nested. Test predictions agree 98.1% with `preds_final/t2_nb.csv`.
  Expected LB change for T2 ≈ +0.001 to +0.002 vs the honest baseline (tiny, but real and not overfit).

**ID-leak variant (team decision):** `idleak_oof_proba.npy`, `idleak_test_proba.npy`, `t2_idleak_test.csv`.
CV 0.9677 ± 0.0000. Reproduce everything with `python t123/research/t2/make_final.py`.
Experiments cache to `t123/research/t2/cache/` (git-ignored).

## 4. Stopping (round 1)
The stop rule fired: more than 6 consecutive ideas (E01, E02, E04×3, E05×2, E07, E08×2, E09) gave no gain ≥ 0.001.

## 5. Round 2 ("research more")
Added E11–E14 and the flip / parametric / T3 / sharp-edge checks. Kept: E14 g2 + EM (+0.0012 honest).
The feature-only track is now within ≈ 0.001–0.002 of its learning-curve ceiling. The ID track is at its ceiling (flips are random).

## 6. Decision: use the leak for T2 (user request)
* The submission T2 = `t2_idleak_test.csv` (player_id block rule). CV macro-F1 0.9677, the ceiling: P(label | id block) is
  0.97 / 0.01 / 0.01 / 0.01, so argmax is macro-F1-optimal and nothing (features included) can fix the random 3% flips.
* `swap_t2.py <sub_v6.csv> <sub_v7.csv>` puts it into an existing 5-task file (checks ids / row count). It changes about 21% of T2 rows vs `t2_nb.csv`.
* Expected public-LB change ≈ w_T2 × (0.968 − 0.756) ≈ +0.05 if w_T2 ≈ 0.25.
* Disclose in the report that T2 uses the player_id ordering (a data-generation artefact). No test rows are used for training.

## 7. Can T2 go above 0.968? (user asked for 0.98)
No, not on noisy labels. A LightGBM "flip detector" on all 44 columns + id + position in block + NB p(block class)
reaches AUC 0.4992 (chance). NB's p(block class) alone gives AUC 0.506. The flipped label's rank among the 3 wrong
classes by NB is 0.344 / 0.333 / 0.323 (uniform). The flips are generated independently of everything in the data,
so the CV ceiling is 0.968. The only unknown is whether Kaggle's *test* labels also carry the flips. If they are clean,
the block rule already scores ≈ 1.0 there.
