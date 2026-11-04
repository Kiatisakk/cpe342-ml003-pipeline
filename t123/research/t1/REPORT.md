# T1 (cheater detection): research report

Branch `research/t1`. Everything lives in `t123/research/t1/`. Every script runs from the repo root (`python t123/research/t1/<script>.py`).
Nothing was submitted to Kaggle. No pretrained model, AutoML tool or test row was used for training.

## Result

| | AUC | F2-macro (OOF-opt thr) | **F2-macro nested thr** | binary F2 (OOF-opt) | binary F2 nested |
|---|---|---|---|---|---|
| Baseline `oof_blend.npy` (XGB+LGB+Cat, Optuna, rank blend), seed-42 folds | 0.89248 | 0.78026 | 0.77987 | 0.83834 | 0.83819 |
| **Final** (seed-42 folds, same scheme as the baseline) | **0.89403** | **0.78225** | **0.78233** | **0.83928** | **0.83895** |
| **Final**, mean ± sd over fold-split seeds 42 / 7 / 2024 | 0.89412 ± 0.00008 | 0.78225 ± 0.00026 | **0.78184 ± 0.00064** | 0.83904 ± 0.00029 | 0.83877 ± 0.00016 |

Honest gain: **about +0.002 F2-macro** (nested 0.7799 → 0.7818) and **+0.0016 AUC**. Binary F2 rises from 0.8382 to 0.8388 (nested).
The gain is about 3 seed-sd, so it is real but small. The F2-macro ceiling of this data looks close to 0.78.

"Nested thr" means the threshold, stored as a positive rate, is picked on the other 4 folds' OOF and applied to the held-out fold.
The final model has no fitted blend weights or calibration, so nothing else needs nesting.

### Final recipe (`final.py`)
1. **Features `fe_v2`**: the baseline feature set from `task1_gbm_pipeline.py` (15 signal columns, the noise 16 dropped, missing flags, lps-identity bounds, zone flags, ratios, frequency encodings), plus the **exact zone posteriors** from exp03.
   - When `account_age_days` is missing (14.5%), use the identity `lps = level·r/age` with r ~ U(0.15, 1.08) and the empirical age prior. This gives the posterior over age on a 1-day grid: posterior ∝ prior(age) · age · 1[r ∈ range].
   - From that posterior: P(age in each class-exclusive zone) and E[age]. The zones are cheater-only 110–120 d, and legit-only <15 d, 180–200 d and ≥1000 d. Two of them (110–120 and 180–200) were not among the baseline's zone flags.
   - The same is done for a missing `level`: P(level<28) for cheater-only, P(level>96) for legit-only, and E[level].
   - The features are label-free. The prior is fitted on train rows only (labeled and unlabeled); test is only transformed.
2. **Model**: LightGBM with the baseline's Optuna-tuned params, **lr 0.02** (was 0.05 in screening), early stopping on an inner 10% split, as in the baseline.
3. **Bagging**: 3 bags per fold, each with a different inner split and model seed, averaged.
4. **CV**: 5-fold StratifiedKFold, seed 42 (plus 7 and 2024 for the error bars). `best_oof.npy` holds the seed-42 OOF probabilities in labeled-train file order (90,816 rows).
5. **Test**: the average of 15 models (5 folds × 3 split seeds) × 3 bags. The test cut uses the OOF-optimal positive rate, averaged over seeds:
   - macro: 42.3% predicted cheaters → `t1_best_test.csv`
   - binary F2: 63.1% predicted cheaters → `t1_best_test_binaryF2.csv`

   A raw probability cut would give the same rate for macro (42.3%) but only 60.1% for binary F2, because averaging smooths the low-probability tail. That is why the rate cut is used.

   Test predictions agree with the baseline's `t1_blend.csv` on 97.3% of rows (Spearman 0.993).

Run time is about 7 minutes on 4 CPU cores.

## Every experiment

Screening uses one LightGBM with the baseline params at lr 0.05, unless the row says otherwise. Rows marked "3 seeds" are mean ± sd over fold splits 42 / 7 / 2024; the others use seed 42 only.
Raw numbers are in `out/results.jsonl` (single seed) and `out/results_ms.jsonl` (multi-seed). Logs are in `out/*.log`.

| # | experiment | AUC | F2m | F2m nested | bin F2 | kept? | note |
|---|---|---|---|---|---|---|---|
| 00 | baseline blend (repo OOF) re-scored | 0.89248 | 0.78026 | 0.77987 | 0.83834 | ref | harness matches the stated 0.7802 |
| 00 | baseline single models (repo) xgb / lgb / cat | .8914/.8919/.8912 | .7790/.7794/.7790 | .7780/.7784/.7779 | .8377/.8375/.8373 | ref | |
| 01 | raw 15 signal columns | 0.89166 | 0.77959 | 0.77903 | 0.83687 | – | trees alone recover most of it |
| 01 | raw 31 columns (with the noise 16) | 0.89129 | 0.78011 | 0.77998 | 0.83709 | – | |
| 01/02 | baseline FE (my re-implementation), 3 seeds | 0.89244±.0003 | 0.78072 | 0.78006±.0004 | 0.83776 | ref | one LGB ≈ the baseline 3-model blend |
| 02 | baseline FE + noise 16, 3 seeds | 0.89202±.0004 | 0.78068 | 0.78002±.0007 | 0.83746 | ✗ | re-check of the "noise" columns, see below |
| **03** | **fe_v2 = + exact zone posteriors for missing age/level**, 3 seeds | **0.89315±.0002** | 0.78128 | **0.78108±.0007** | 0.83840 | ✓ | +0.0007 AUC on every seed |
| 04 | mixture naive Bayes (EM, from scratch), K = 1/3/6/8/12 components | .848/.867/.875/.873/.871 | – | .743–.763 | – | ✗ | Bayes-style model, NaN marginalised; below the GBMs |
| 07 | mixture NB K=(6,3), 3 seeds | 0.86957 | 0.76243 | 0.76215 | 0.82083 | ✗ | |
| 07 | mixture NB + 13k **unlabeled train** rows in EM (semi-supervised), 3 seeds | 0.86847 | 0.76170 | 0.76119 | 0.81966 | ✗ | the unlabeled rows do not help |
| **05** | **fe_v2, LGB lr 0.02**, 3 seeds | **0.89374±.0001** | 0.78179 | **0.78151±.0006** | 0.83890 | ✓ | |
| 05 | fe_v2, scale_pos_weight 1.86 (cost-sensitive), 3 seeds | 0.89356 | 0.78173 | 0.78150 | 0.83881 | ✗ | the threshold already handles F2 |
| 05 | fe_v2, bag of 5 (inner split + seed), lr 0.05, 3 seeds | 0.89405 | 0.78208 | 0.78133±.0008 | 0.83908 | ~ | AUC up, F2 the same; the idea goes into the final at lr 0.02 |
| 06 | XGB on fe_v2, 3 seeds | 0.89279 | 0.78091 | 0.78054±.0005 | 0.83864 | – | baseline XGB was 0.7780, so the zone features carry over |
| 06 | CatBoost on fe_v2, 3 seeds | 0.89235 | 0.77968 | 0.77925±.0012 | 0.83806 | – | |
| 09 | MLP from scratch (PyTorch, piecewise-linear encoding, 3×256, early stop), 3 seeds | 0.88548 | 0.77281 | 0.77177±.0011 | 0.83473 | ✗ (alone) | |
| B | nested blend LGB + MLP + NB: rank-avg / AUC-weights / LR-stack | .8906/.8938/.8938 | | .7794/.7812/.7810 | | ✗ | weights: MLP 0.08, NB 0.00 |
| B | nested blend LGB + XGB + Cat: rank-avg / AUC-weights / LR-stack | .8938/.8939/.8939 | | .7809/.7814/.7811 | | ✗ | ≤ LGB lr 0.02 alone |
| B | nested blend LGB-bag5 + LGB lr0.02 + XGB: rank / weights / LR | .8940/.8941/.8941 | | .7818/.7812/.7817 | | ~ | same as LGB-bag alone; LR stacking ≈ rank blend |
| 08 | fe_v2 + game_sense/accuracy (course hint), 3 seeds | 0.89332 | 0.78173 | 0.78117 | 0.83855 | ✗ | +0.0001 = noise |
| 08 | fe_v2 + 45 pairwise log-ratios of skill features + 3 others (104 feats), 3 seeds | 0.89337 | 0.78155 | 0.78114 | 0.83857 | ✗ | noise |
| 10 | Optuna re-tune of LGB on fe_v2 (60 trials, separate 3-fold seed 99), eval lr 0.02, 3 seeds | 0.89342 | 0.78119 | 0.78042 | 0.83921 | ✗ | +0.0005 on the tuning split, does not transfer |
| 11 | in-fold pseudo-labels on the 13k **unlabeled train** rows (conf 0.15, weight 0.5), 3 seeds | 0.89329 | 0.78101 | 0.78051 | 0.83855 | ✗ | vs 0.78108 without |
| F1 | final v1: LGB lr0.02 bag3 + XGB, equal average, 3 seeds | 0.89386 | 0.78210 | 0.78167±.0004 | 0.83907 | ✗ | XGB half dilutes the LGB (`out/final_v1_lgb_xgb.log`) |
| **F** | **final: LGB lr 0.02, bag 3, fe_v2**, 3 seeds | **0.89412±.0001** | **0.78225** | **0.78184±.0006** | **0.83904** | ✓ | |

After the last kept change (lr 0.02), the next 8 ideas gave nothing beyond noise (class weights, CatBoost/XGB blending, MLP, NB, semi-supervised ×2, ratios, Optuna), so the loop stopped under the 6-idea rule.

## Findings worth keeping
- **The "noise" 16 features are not pure noise, but they carry no class information.** Each has marginal AUC ≈ 0.50 and KS p > 0.07. Several correlate with signal features (input_consistency–reaction_time ρ≈0.26, damage–win_rate, first_blood–reaction_time), equally in both classes. Adding them costs 0.0004 AUC, so dropping them is right.
- **Legit players are a mixture of age-based sub-populations; cheaters are two populations, age 15–180 and 200–1000.**
  - Legit "new" accounts (age < 110) have friends 0–50, reports 10–29 and higher KD.
  - Legit accounts aged 120–1000 have friends 50–240 and reports 5–19.
  - That leaves sharp class-exclusive age windows: cheater-only 110–120 d; legit-only <15 d, 180–200 d and >1000 d.
  - Trees find these windows when age is observed. When age is missing, the identity-based posterior recovers part of that information, and that is where the exp03 gain comes from.
- The level-speed ratio r = lps·age/level is U(0.15, 1.08) in both classes and in every age segment, so it is not informative by itself.
- The from-scratch MLP (0.8855) and mixture NB (0.875) are clearly worse than the GBMs and get ~0 blend weight. Within-class feature correlations (aim/spray/reaction ≈ 0.3) break the NB assumptions.
- Unlabeled train rows: neither semi-supervised EM nor pseudo-labeling helps. With 90k labels the extra 13k rows add little.

## Files
- `common.py`: data loading, baseline FE re-implementation, `fe_zone_post` / `fe_v2`, vectorised F-beta threshold search, nested threshold scoring, CV runners for LGB/XGB/CatBoost, multi-seed reporting
- `exp00…exp11*.py`: one script per experiment (table above); `mixnb.py` holds the mixture naive Bayes; `blend.py` does nested blending (rank / AUC weights / LR stacking)
- `final.py`: final recipe. It writes:
  - `best_oof.npy`: seed-42 OOF probabilities
  - `best_oof_allseeds.npy`: (3, 90816)
  - `best_test_proba.npy`
  - `t1_best_test.csv` and `t1_best_test_binaryF2.csv` (`id,pred,score`, ids ANS00001..ANS25889 in test order)
  - `final_results.json`
- `out/`: logs and result jsonl files. OOF `.npy` files and the Optuna db are git-ignored and can be regenerated.

Not done (left for later): out-of-fold kNN features, monotone constraints. Monotone constraints are unlikely to help, because the class signal is zone-shaped rather than monotone.
