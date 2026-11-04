# T1 (cheater detection): research report

Branch `research/t1`. Everything lives in `t123/research/t1/`. Every script runs from the repo root (`python t123/research/t1/<script>.py`).
Nothing was submitted to Kaggle. No pretrained model, AutoML tool or test row was used for training.

## Result

| | AUC | F2-macro (OOF-opt thr) | **F2-macro nested thr** | binary F2 (OOF-opt) | binary F2 nested |
|---|---|---|---|---|---|
| Baseline `oof_blend.npy` (XGB+LGB+Cat, Optuna, rank blend), seed-42 folds | 0.89248 | 0.78026 | 0.77987 | 0.83834 | 0.83819 |
| Round 1 final (`final.py`, now writes `v1_*`), 3 seeds | 0.89412 ± 0.00008 | 0.78225 ± 0.00026 | 0.78184 ± 0.00064 | 0.83904 ± 0.00029 | 0.83877 ± 0.00016 |
| **Final v2** (`final_v2.py`), seed-42 folds (baseline scheme) | **0.89446** | **0.78243** | **0.78186** | **0.83955** | **0.83922** |
| **Final v2**, mean ± sd over fold-split seeds 42 / 7 / 2024 | **0.89453 ± 0.00006** | 0.78251 ± 0.00034 | **0.78201 ± 0.00046** | 0.83938 ± 0.00021 | 0.83911 ± 0.00026 |
| **Final v2, 3-seed averaged OOF** (= the model used for test; cut from this OOF, exp17) | **0.89482** | 0.78310 | **0.78279 ± 0.00012** | 0.83948 | 0.83911 ± 0.00015 |

Honest gain over the baseline: **about +0.003 F2-macro** for the delivered test model (nested 0.7799 → 0.7828 with the 3-seed averaged OOF; 0.7820 for a single split), **+0.002 AUC** (0.8925 → 0.8945), and **+0.001 binary F2** (nested 0.8382 → 0.8391).

Round 2 (MCAR masking augmentation) improved on round 1 as follows:
- AUC +0.0004, which is solid at 7× the seed sd.
- Binary F2 +0.0003.
- F2-macro only +0.0002. By the "< 0.001 is noise" rule, that part is not a real macro gain.

v2 was adopted because it is at least as good as round 1 on every 3-seed mean, with lower variance. The F2-macro ceiling of this data looks close to 0.78.

"Nested thr" means the threshold, stored as a positive rate, is picked on the other 4 folds' OOF and applied to the held-out fold.
The final model has no fitted blend weights or calibration, so nothing else needs nesting.

### Final recipe (`final_v2.py`, about 26 minutes on 4 CPU cores)
1. **Features `fe_v2`**: the baseline feature set from `task1_gbm_pipeline.py` (15 signal columns, the noise 16 dropped, missing flags, lps-identity bounds, zone flags, ratios, frequency encodings), plus the **exact zone posteriors** from exp03.
   - When `account_age_days` is missing (14.5%), use the identity `lps = level·r/age` with r ~ U(0.15, 1.08) and the empirical age prior. This gives the posterior over age on a 1-day grid: posterior ∝ prior(age) · age · 1[r ∈ range].
   - From that posterior: P(age in each class-exclusive zone) and E[age]. The zones are cheater-only 110–120 d, and legit-only <15 d, 180–200 d and ≥1000 d. Two of them (110–120 and 180–200) were not among the baseline's zone flags.
   - The same is done for a missing `level`: P(level<28) for cheater-only, P(level>96) for legit-only, and E[level].
   - The features are label-free. The prior is fitted on train rows only (labeled and unlabeled); test is only transformed.
2. **MCAR masking augmentation** (exp16).
   - Missingness is MCAR: the cheater rate given NaN equals the base rate for every column.
   - In each fold, 4 copies of the fold-train rows get extra NaNs. Each signal value is hidden with probability equal to its observed missing rate, and `fe_v2` is recomputed on the masked raw data.
   - The model therefore learns P(y | x without reports/friends/age/…) from all 72k fold-train rows instead of only the ~16k that are really incomplete. Those incomplete rows are where AUC drops from 0.90 to 0.855.
   - The early-stopping holdout, the held-out fold and test are never augmented.
3. **Model**: LightGBM with the baseline's Optuna-tuned params, lr 0.02, early stopping on an inner 10% split as in the baseline. 3 bags per fold (different inner split and seed, and different masks), averaged.
4. **CV**: 5-fold StratifiedKFold, seed 42 (plus 7 and 2024 for the error bars). `best_oof.npy` holds the seed-42 OOF probabilities in labeled-train file order (90,816 rows).
5. **Test**: the average of 5 folds × 3 split seeds × 3 bags. The matching OOF is the **average of the 3 split-seed OOFs** (`best_oof_avg3.npy`): each row's 3 predictions come from models that never saw it.
   The test cut is the optimal positive rate of that averaged OOF (`make_test_files.py`, round 3):
   - macro: 41.25% predicted cheaters → `t1_best_test.csv`, nested F2m 0.7828 ± 0.0001
   - binary F2: 63.0% predicted cheaters → `t1_best_test_binaryF2.csv`, nested 0.8391

   A raw probability cut would give the same rate for macro but too few positives for binary F2, because averaging smooths the low-probability tail. That is why the rate cut is used.

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
| F | round-1 final: LGB lr 0.02, bag 3, fe_v2, 3 seeds (`final.py`, now writes `v1_*`) | 0.89412±.0001 | 0.78225 | 0.78184±.0006 | 0.83904 | ✓ (round 1) | |
| **— round 2 —** | | | | | | | |
| 12 | label-free LGB imputers (zone probs + E[value] for missing reports/friends/crosshair/age/level, from the 30 other raw cols incl. noise), 3 seeds | 0.89306 | 0.78129 | 0.78105±.0002 | 0.83860 | ✗ | imputed crosshair predicts y at AUC 0.77 on its missing rows, but only by proxying class from the other columns, which the GBM already sees |
| 13 | LGB extra_trees / linear_tree / GOSS / deep+reg / shallow, 3 seeds each | .8918/.8930/.8909/.8914/.8927 | | .7801/.7802/.7784/.7789/.7799 | | ✗ | all below the default; nested blend with lgb lr0.02 gives 0.7813 (no gain) |
| 14 | nested out-of-fold kNN target rates (k 25/100/400, rank-gauss space) as features, seeds 42 & 7 | .8926/.8924 | | .7799/.7795 | | ✗ | kNN alone AUC 0.795; worse than without (stopped after 2 seeds) |
| 15 | ablation: drop freq/log features / skill-z + interactions / both ("compact", 34 feats), 3 seeds | .8933/.8934/.8937 | | .7815/.7805/.7811 | | ✗ | compact gives +0.0005 AUC but no F2 gain |
| **16** | **MCAR masking augmentation, 2 copies, rate 1×**, lr 0.05, 3 seeds | 0.89376±.0002 | 0.78205 | 0.78144±.0010 | 0.83876 | ✓ | +0.0006 AUC over exp03 on every seed |
| **16** | **4 copies, rate 1×**, lr 0.05, 3 seeds | **0.89386±.00002** | 0.78202 | **0.78176±.0006** | 0.83890 | ✓ | rows missing reports/friends/age/crosshair: +0.001–0.002 AUC |
| 16 | 2 copies, rate 2×, lr 0.05, 3 seeds | 0.89372 | 0.78156 | 0.78115 | 0.83877 | ✗ | over-masking is no better |
| **F2** | **final v2: fe_v2 + 4 masked copies + LGB lr 0.02 bag 3**, 3 seeds | **0.89453±.00006** | **0.78251** | **0.78201±.0005** | **0.83938** | ✓ | AUC +0.0004 vs round-1 final (solid); F2m +0.0002 (noise) |
| **— round 3: post-processing (exp17, on the saved OOF; every tuned quantity nested) —** | | | | | | | |
| 17A | single positive-rate cut (what final_v2 does) vs single probability cut | | | 0.78201 / 0.78201 | | ref | identical |
| 17B | hard overrides in observed class-exclusive zones (6,658 forced to 1, 24,397 to 0) | | | 0.78202 | | ✗ | the model already gets all 31,055 zone rows right |
| 17C | per-group thresholds, jointly tuned (groups: 0/1/2+ key features missing; or reports/friends missing) | | | 0.78166±.0011 / 0.78202±.0007 | | ✗ | more variance, no gain: a single cut on the posterior is already optimal for F-measures |
| 17D | per-group isotonic calibration + one probability cut (same groups) | | | 0.78146±.0013 / 0.78180±.0006 | | ✗ | |
| **17E** | **average of the 3 split-seed OOFs (= the averaged test model), rate cut from it** | **0.89482** | 0.78310 | **0.78279±.0001** | 0.83948 | ✓ | +0.0008 on every seed; really a bigger ensemble, not post-processing. Test cut now set from this OOF (41.25% vs 40.5%) |

Round 1: after the last kept change (lr 0.02), the next 8 ideas gave nothing beyond noise (class weights, CatBoost/XGB blending, MLP, NB, semi-supervised ×2, ratios, Optuna), so the loop stopped under the 6-idea rule.

Round 2: error analysis showed the error is concentrated in rows missing reports/friends/age/crosshair (AUC 0.854–0.863 vs 0.90). Three missing-value ideas followed: imputers ✗, kNN ✗, masking augmentation ✓. After the augmentation, the rest (LGB variants, rate 2×, feature ablation) gave nothing beyond noise.

## Findings worth keeping
- **The "noise" 16 features are not pure noise, but they carry no class information.** Each has marginal AUC ≈ 0.50 and KS p > 0.07. Several correlate with signal features (input_consistency–reaction_time ρ≈0.26, damage–win_rate, first_blood–reaction_time), equally in both classes. Adding them costs 0.0004 AUC, so dropping them is right.
- **Legit players are a mixture of age-based sub-populations; cheaters are two populations, age 15–180 and 200–1000.**
  - Legit "new" accounts (age < 110) have friends 0–50, reports 10–29 and higher KD.
  - Legit accounts aged 120–1000 have friends 50–240 and reports 5–19.
  - That leaves sharp class-exclusive age windows: cheater-only 110–120 d; legit-only <15 d, 180–200 d and >1000 d.
  - Trees find these windows when age is observed. When age is missing, the identity-based posterior recovers part of that information, and that is where the exp03 gain comes from.
- The level-speed ratio r = lps·age/level is U(0.15, 1.08) in both classes and in every age segment, so it is not informative by itself.
- The from-scratch MLP (0.8855) and mixture NB (0.875) are clearly worse than the GBMs and get ~0 blend weight. Within-class feature correlations (aim/spray/reaction ≈ 0.3) break the NB assumptions.
- **Where the error is**: AUC is 0.918 for rows with no missing signal feature and falls to 0.81 with 5 missing. Rows missing reports, friends, age or crosshair score 0.854–0.863. No ID or row-order leakage (player_id AUC 0.498; train/test IDs disjoint); no duplicates; no value-precision artefacts.
- **MCAR is exploitable**: because missingness is independent of the label, artificially masking training rows is a valid augmentation. It is the only missing-value idea that helped beyond the exact identity posterior.
- Unlabeled train rows: neither semi-supervised EM nor pseudo-labeling helps. With 90k labels the extra 13k rows add little.

## Files
- `common.py`: data loading, baseline FE re-implementation, `fe_zone_post` / `fe_v2`, vectorised F-beta threshold search, nested threshold scoring, CV runners for LGB/XGB/CatBoost, multi-seed reporting
- `exp00…exp16*.py`: one script per experiment (table above); `mixnb.py` holds the mixture naive Bayes; `blend.py` does nested blending (rank / AUC weights / LR stacking)
- `make_test_files.py`: rebuilds the test CSVs from the saved predictions, with cuts from the averaged OOF; `final_v2.py` calls it at the end
- `exp17_postproc.py`: the post-processing tests (round 3)
- `final_v2.py`: **final recipe**. It writes:
  - `best_oof.npy`: seed-42 OOF probabilities
  - `best_oof_allseeds.npy`: (3, 90816)
  - `best_test_proba.npy`
  - `t1_best_test.csv` and `t1_best_test_binaryF2.csv` (`id,pred,score`, ids ANS00001..ANS25889 in test order)
  - `final_results.json`
- `final.py`: round-1 recipe (no augmentation); writes the `v1_*` copies of the same files
- `out/`: logs and result jsonl files. OOF `.npy` files and the Optuna db are git-ignored and can be regenerated.

Not done: monotone constraints. They are unlikely to help, because the class signal is zone-shaped rather than monotone.
Possible next steps: combine the compact feature set with augmentation; more bags or seeds for the test prediction (variance only).
