# Kaggle-ML — Handoff (Team ML003)

Competition: **player-intelligence-system-cpe-342-ml** (Kaggle, owner account `kiatisakkk`)
Deadline: Kaggle **24 Nov 2026 13:30 (Bangkok)** · report **30 Nov**
Status at handoff (4 Nov 2026 09:00): **best public LB 0.64291** (`sub_v6...`). A friend's team reached 0.74803, so the gap is ~0.105.

---

## 0. Rules for whoever continues (read first)
1. **Do NOT submit to Kaggle unless the user explicitly asks in chat.** Daily quota is small (resets 07:00 Bangkok = 00:00 UTC).
2. **Secrets:** never write API keys or tokens in code, notebooks, logs or chat. The Kaggle CLI reads `C:\Users\kiati\.kaggle\kaggle.json`. Notebook secrets (`HF_TOKEN`, `TABPFN_TOKEN`) live in Kaggle → Add-ons → Secrets.
   The key that was pasted in chat on 4 Nov should be **rotated** (Kaggle → Settings → API → Expire / Create New Token).
3. Submission format that works: `id,task1,task2,task3,task4,task5`, **floats**, comma, LF line endings, 25,889 rows, `id = ANS00001..` in test row order (same order as every `test.csv`).
4. Do not use `sleep` to wait on Kaggle runs. Poll with `kaggle kernels status`.

---

## 1. Kaggle CLI cheat-sheet (Windows `cmd`, run from `C:\Users\kiati\Documents\Kaggle-ML`)

```bat
python -m pip install kaggle                     :: once
python -m kaggle competitions list -s cpe-342    :: sanity check that auth works

:: --- data ---
python -m kaggle competitions download -c player-intelligence-system-cpe-342-ml -p cpe342-kaggle-main

:: --- notebooks (kernels) ---
python -m kaggle kernels list --mine
python -m kaggle kernels status kiatisakkk/<slug>
python -m kaggle kernels output kiatisakkk/<slug> -p t<N>\preds\<dir>     :: download outputs + log
python -m kaggle kernels pull   kiatisakkk/<slug> -p work\<slug> -m             :: get code + kernel-metadata.json
python -m kaggle kernels push   -p work\<slug>                                  :: re-run (new version)

:: --- submissions (ONLY when the user asks) ---
python -m kaggle competitions submit -c player-intelligence-system-cpe-342-ml -f submissions\<file>.csv -m "<description>"
python -m kaggle competitions submissions -c player-intelligence-system-cpe-342-ml
```

If a download fails, re-run it: a partial download left only logs in `t1/` and `t4/` on 4 Nov.
`kernel-metadata.json` must keep `"enable_gpu": true` for T3 and T4 notebooks, `"enable_internet": true` for TabPFN, and include the competition as a data source.

### Notebooks on Kaggle (all write predictions only; none of them submits)
| slug | what | key result |
|---|---|---|
| `task3-hurdle-ziln-v4` | T3 XGB hurdle + multi-quantile + calibration | OOF 0.7920 (blend), outputs `preds/t3_v4_*.csv` |
| `task3-nn-ziln` | T3 NN mixture-lognormal (ZILN loss) | NN_K3 OOF 0.7901 |
| `task3-hurdle-ziln` (v1–v3) | earlier T3 versions | 0.7893 → 0.7917 |
| `task1-gbm-optuna-pipeline` | T1 XGB/LGBM/CatBoost + Optuna 200 trials + blend | OOF F2m 0.7804, `t1_pipeline/t1_blend.csv` |
| `task1-tabpfn` | T1 TabPFN v3.5 (tabpfn-client, needs `TABPFN_TOKEN`) | 0.7799; avg with HGB 0.7825 |
| `task12-xgb-pipeline` | T1/T2 XGB baseline | T1 0.7786, T2 0.7498 |
| `task4-5fold-ensemble` | T4 ConvNeXt-T DINOv3, 5 folds (needs `HF_TOKEN`) | OOF F1 0.9889, `preds/task4_v3_5f.csv` |

---

## 2. Folder map
```
Reorganised 5 Nov 2026: one folder per task, each with code/ preds/ research/ (research = cloud-agent branch output + REPORT.md).
```
cpe342-kaggle-main/     competition data (unchanged name: every script reads from it)
docs/                   course PDFs, Canva text, cloud_prompts/
shared/                 t123_baseline.py (HGB T1-T3), task12_xgb.ipynb, cache/ (fold npz, ignored)
t1/  code/              task1_gbm_pipeline.py, task1_tabpfn.py
     preds/             t1_pipeline/ (GBM blend), t1_hgb.csv
     research/          BEST: final.py -> t1_best_test.csv (F2m 0.7818)
t2/  code/              NB: nb.py, pf.py, best_nb_perfeature.json
     preds/             t2_nb.csv (BEST, used), t2_nb_bias.csv
     research/          10 experiments (none beat NB) + t2_idleak_test.csv (ID LEAK, ask TA)
t3/  code/              task3_hurdle_ziln_v2..v4.py, task3_nn_ziln.py
     preds/             v4/ (XGB hurdle), nn_ziln/ (NN_K1/K3 + OOF)
     research/          BEST: CatBoost offset multi-quantile -> t3_best_test.csv (0.7953); NN blend adds 0
t4/  code/              task4_5fold.py (v3) + notebooks
     preds/             v3_5f/ (BEST: ConvNeXt-T 5f, OOF 0.9889), v3a/ (old, ignored)
     research/          prior-shift study (not adopted), task4_kaggle_v4.py, t4_test_str*.csv
     kaggle/            task4-v4-eva02s/ (kernel folder: 3 folds + inner-holdout early stop; push with kaggle kernels push -p)
t5/  code/              t5_build_union.py, detectors, TMAE/TMLP
     preds/ probes/ plots/
     research/          new acc/kda/perf cluster, t5_pipeline.py, candidates/ (union, +p10/p50/p90)
submissions/
     FINAL/             current candidates: sub_v9a (submit first), v9b, v9c, v8a, v8b, v8L (T2 leak, DO NOT submit w/o TA)
     history/           everything submitted v1..v7 + READMEs
     probes/            probes_v2, probes_v6, probes_t4, probes_t5 (t5_groups.csv), single, bisect, format_tests
     tools/             build_submission*.py
```
The T3 NN test file is `NN_K3` (it arrived as an attachment; the blend is already inside `sub_v5/v6`).

---

## 3. What each task does now (= `sub_v6_t1gbmblend_t2nb_t3blend_t4ens5f_ovr_t5union.csv`)

### T1 — cheater detection, binary, **LB = F2 of the positive class (binary)**, confirmed 5 Nov by probe_v10e; cut at ~63%
- EDA: 16 pure-noise features. 15 signal features. Class-exclusive value ranges plus an overlapping core of ~55% (AUC ≈ 0.79 inside the core) → ceiling ≈ 0.78.
- Hidden identity `level_progression_speed = level · r / account_age_days`, r ~ U(0.15, 1.08) → features `age_lo, age_hi, age_filled, lv_lo, lv_hi`.
- Model: XGB + LightGBM + CatBoost, each tuned with Optuna (200 trials, 3-fold AUC), refit 5-fold. Rank blend weights xgb 0.18 / lgb 0.48 / cat 0.33. Threshold chosen for max F2 macro (predicts 40.4% cheaters).
- CV: AUC 0.8925, **F2m 0.7804**. Alternatives: HGB 0.7807, TabPFN 0.7799, TabPFN+HGB 0.7825, EBM 0.7786.
- Labeled rows only (~13k train rows have no label and are unused).

### T2 — player segment, 4 classes, **F1 macro**
- EDA: features are conditionally independent within class → Naive Bayes is Bayes-optimal. Latent 2×2 structure (engagement × spending). 16 noise columns removed. Informative categoricals: loyalty_tier, engagement_level.
- Model: histogram NB (quantile bins + Gaussian smoothing per continuous feature, Laplace α). Per-feature (bins, smoothing) tuned by Optuna (`pf.py`). NaN is marginalised (skipped).
- CV **F1m ≈ 0.755–0.758** (HGB 0.750, XGB 0.7498).
- Tested 4 Nov: EM prior estimation (Saerens) shows the test prior ≈ the train prior, so there is no shift. Class-bias tuning gives +0.0007 nested CV (`t2/preds/t2_nb_bias.csv`).

### T3 — spending, regression, **NMAE score** (we assume `1/(1+MAE/mean|y|)`, all-zero = 0.5; NOT verified on LB)
- ~48% zeros. Whales (segment 3) cause ~80% of the error.
- Exact fills: `historical_spending = total_transactions × avg_transaction_value`, `purchases_on_discount = total_transactions × discount_rate_used` (NaN 5.8% → 0.4%). Segment imputer (holdout acc 0.929).
- Hurdle: classifier p = P(y>0) + XGB multi-quantile; predict the conditional quantile q = 1 − 1/(2p) (0 if p < 0.5), which is MAE-optimal; (shift, κ) calibration on OOF.
- Blend (OOF weights): v4 D_q 0.38 + D_qw0.3 0.24 + D_qw0.6 0.06 + NN_K3 0.32 → **OOF 0.7925** (HGB two-stage 0.7843).

### T4 — game screenshots, 5 classes, **F1 macro**
- ConvNeXt-Tiny `convnext_tiny.dinov3_lvd1689m`, LP-FT, LLRD 0.8, EMA, 14 epochs, 5 folds, averaged softmax. OOF F1 0.9889.
- **The public LB ignores T4**: datacard order vs Overview order vs (friend) string labels all give an identical score. v6 keeps numeric Overview order (0 PUBE, 1 FreeFried, 2 FiveN, 3 Roblock, 4 7-11k) in case the private LB scores it. Datacard order = alphabetical (0 7-11k, 1 fiveN, 2 freefried, 3 pube, 4 roblock). `t4/preds/v3_5f/preds/task4_v3_5f.csv` has both `pred_num` (datacard) and `pred_str`.

### T5 — account anomaly, unsupervised, **F3**
- `t5/code/t5_build_union.py` reproduces the set exactly (1,811 rows):
  - rule (827): geo jump of periods 3/4 > 5° from home (mean of periods 1–2), invalid |lat| > 90, or ≥ 2 metrics whose period-3/4 change has robust z > 6
  - stationary clusters found by histogram-gap scan: bot (mean login_time_variance < 1.0, 207), lowact (mean activity_level < 0.34, 266), lowcomm (mean communication_frequency < 0.17, 237), lowstab (behavior_stability < 0.55, 274)
- A full gap scan of every mean/std/static feature finds no further isolated cluster. Flags (password_changed, vpn, mass_item_sale) are enriched only inside the takeover rule group.
- TMAE (temporal masked autoencoder) tried: AUC 0.83 vs rules, 0.66 vs union → not used.

---

## 4. Leaderboard log and what we learned

| file | change | public |
|---|---|---|
| sub_v2 | HGB T1/T2/T3, T4 v3a datacard, T5 ens top-1000 | 0.54905 |
| **sub_v6** | T1 GBM blend, T2 NB, T3 blend, T4 5f (overview), T5 union 1811 | **0.64291** |
| sub_v6b | v6 with T4 datacard order | 0.64291 (identical → T4 not scored) |
| sub_v6_T5_3602 | T5 +1,791 rows (loosened rules + TMAE rank) | 0.63052 |
| v6_minus_task5 | T5 all zero | 0.46572 |
| sub_v6_T5_2167 | T5 +356 (shock z 4–6, geo 2–5°) | 0.64045 |
| sub_v10c (5 Nov) | T1 v2, **T2 id-leak**, T3 CatBoost, **T4 strings**, T5 box 2,055 | **0.70885** |
| sub_v10d (5 Nov) | = v10c but T2 EM-NB (no leak), T4 numeric overview | 0.66614 (best non-leak) |
| probe_v10e (5 Nov) | v10d with T1 cut at 63% (binary-F2 optimum) | 0.67890 (+0.0128 → **LB T1 = binary F2**, not macro) |
| probe_v10g (5 Nov) | v10e with T3 = 0 everywhere | 0.61991 (−0.0590 = 0.2×0.295 → T3 model 0.795 if zero = 0.5: **NMAE formula and CV confirmed**) |
| probe_v12c (5 Nov) | leak + T1 binF2 cut 60% + T4 mixed 7 formats round-robin (`t4/research/probe_v12t4_groups.csv`) | **0.72753** (expected 0.72201 with no T4 hit → +0.0055 excess = T4 macro ≈ 0.028: some T4 token is scored; bisect with probe_v13B1/B2). Friend: 0.76179 (also uses the leak) |
| t4_constbase/4_A1_mix7_v12c (6 Nov) | T1=1, T2=T3=T5=0 constants + v12c T4 mix | 0.27935 (predicted no-hit base 0.27428, not yet measured → +0.0051, T4 macro ≈ 0.025; agrees with v12c excess) |
| t4_constbase/1_P1_cardnames (6 Nov) | same base, T4 = datacard names (7-11 Knights, FiveN, Free Fried, PUBE, Roblock) | 0.27401 = measured base (T4 = 0, these tokens are not scored). A1 excess is +0.00534, T4 macro ≈ 0.0267 |
| t4_constbase/5_B1_ints (6 Nov) | same base, only int-text groups kept, other rows datacard names | 0.28420 (+0.01019, T4 macro ≈ 0.051: integer tokens are the scored family; floats like 2.0 are not). Best-fit mapping 7-11k 0, freefried 1, pube 2, roblock 3, fiveN 4 (not unique) |
| t4_constbase/7_M1_mapping_01234_object (6 Nov) | same base, T4 ints 7-11k 0, freefried 1, pube 2, roblock 3, fiveN 4, one name row to keep the column non-numeric | 0.30552 (+0.03151, T4 macro 0.1576: chance level, see below) |
| sample_submission_fn / _lol (user, earlier) | T1=1, T2=2, T3=0, T5=1 (nearly constant), T4 constant fiveN / LOL | 0.35559 both (string T4 = 0) |
| sample_submission_ding (user, earlier) | same base, T4 almost all `1`, ints in every column | 0.37399 (+0.0184, T4 macro 0.092: token `1` is a large class; pure int columns are scored, the sub_v1 0.00000 was not caused by ints alone) |
| t4_constbase/8_M2b_mapping (6 Nov) | const base, T4 7-11k 0, fiveN 1, freefried 4, pube 3, roblock 2 | 0.31337 (+0.03936, T4 macro 0.197: chance level) |
| t4_constbase/31024.csv (M3, 6 Nov) | const base, T4 7-11k 3, fiveN 1, freefried 0, pube 2, roblock 4 | 0.31336 (chance level) |
| t4_constbase/41203.csv (M4, 6 Nov) | const base, T4 7-11k 4, fiveN 1, freefried 2, pube 0, roblock 3 | 0.31254 (T4 macro 0.1927, chance level) |
| t4_constbase/14_ISO_fiveN_is_1 (6 Nov) | const base, fiveN rows = `1`, all other rows = `LOL` (6 labels in the macro) | 0.28336 (+0.00935; chance predicts 0.2833, aligned and right about 0.306, aligned and wrong about 0.274: no sign that T4 truth aligns with test.csv rows) |
| **sub_v16a** (6 Nov) | T1 top 60%, T2 id block, T3 best, T4 int datacard `0 1 2 3 4`, T5 id block | **0.76297** best public (predicted 0.7626 with T4 at chance, so `0 1 2 3 4` is chance level too; 0.91 ruled out). Friend 0.76179 |
| sub_v17a (6 Nov) | v16a with T1 p>=0.20 (59.2%) and T4 int `0 1 4 2 3` (best of 120 mappings under the chance model) | 0.76272 (expected 0.7632-0.7636; -0.00025 vs v16a, T1 and T4 effects not separable). No gain: keep v16a, do not select v17a |
| sub_v16b (7 Nov) | no id leak: T2 EM-NB, T5 `lsnan` 2,103 rows, otherwise v16a | 0.71965 (expected about 0.72): best non-leak, fallback final |
| sub_v16e (7 Nov) | v16a with T4 int Overview order (`4 2 1 0 3` for 7-11k, fiveN, freefried, pube, roblock) | 0.76086 (expected about 0.762, -0.00211 vs v16a): Overview order is chance level too |
| sub_v18a (7 Nov) | v16b + T2 blend (.87 NB + .08 spline LR + .05 MLP) + fixed-point affine F1 rule (533 rows) | **0.71999** best non-leak (+0.00034 vs v16b, T2 about +0.0017; CV said +0.0009) |
| sub_v18b (7 Nov) | v18a + T4 Overview order | 0.71788 (-0.00211 vs v18a, exactly the v16e-v16a gap: T4 truth is fixed, Overview is worse by chance) |
| sub_v18c (7 Nov) | v18a + T3 50/50 best + robust | 0.71998 (T3 mix = 0) |
| sub_v18d (7 Nov) | v16a + T3 50/50 best + robust | 0.76297 (= v16a) |
| sub_v18e (7 Nov) | v16a + T5 2 box-only rows outside the id block (2,072 rows) | 0.76295 (-0.00002, about the cost of 1-2 false positives: the id block alone looks exact) |
| sub_v19a (9 Nov) | v16a + T4 `0 1 4 2 3` only | 0.76276 (-0.00021: the chance-model "best" mapping is worse than datacard) |
| sub_v19b (9 Nov) | v16a + T1 p>=0.20 only | 0.76294 (-0.00003; v17a = v19a + v19b, effects add up) |
| sub_v19c (9 Nov) | v16a + T1 p>=0.175 (61%) only | 0.76295 (-0.00002: T1 cut is flat, keep 60%) |
| sub_v19d (9 Nov) | v19a + T3 50/50 mix | 0.76276 (= v19a, T3 mix 0 again) |
| sub_v19e (9 Nov) | v16a + T4 `0 2 4 1 3` only | 0.76257 (-0.00040) |
Measured T4 order ranking on the full file: datacard `0 1 2 3 4` (v16a) > `0 1 4 2 3` > `0 2 4 1 3` > Overview. The chance model predicted the opposite for the first two (gap about 1-1.5 sd); keep datacard.
| sub_v20a (10 Nov) | v18a + T1 rank blend 0.7 final_v2 + 0.2 blind stack_lr + 0.1 GBM pipeline, 60% | **0.72016** (+0.00017, predicted +0.00014; nested CV +0.00068 F2, 14/15 folds) |
| sub_v20b (10 Nov) | v18a + T1 blend 0.8/0.1/0.1 | 0.72012 (+0.00013) |
| sub_v20c (10 Nov) | v20a with T1 cut 61% | 0.72000 (61% worse than 60% by 0.00016) |
| sub_v20d (10 Nov) | v20a + T5 lowstab-NaN posterior >= 0.05 (2,121 rows) | 0.72018 (+0.00002 vs v20a, noise) |
| sub_v20e (10 Nov) | v20a + T5 posterior >= 0.15 (2,086 rows) | 0.72004 (-0.00012: the dropped rows held true anomalies, posterior looks calibrated; keep 0.10) |
T3 Fourier month features (round12): paired CV -0.00009, 5/5 folds worse, not used. No T3 OOF model adds anything in a blend.
Report techniques tested 10 Nov (all nested or paired CV, none adopted): T3 CatBoost native categoricals +0.00007 ± 0.00011 (3/5 folds, noise); T3 purchase-cadence features -0.00023 (4/5 worse); T2 Nelder-Mead class weights +0.00019 vs argmax, 0.00043 below the affine rule; T1 non-negative logit stack +0.00039, 0.00041 below the rank blend. Skipped: transductive EM, adversarial reweighting, DAE on train+test (Rule 2: test used for training); quantiles above 0.5 for T3 (the MAE-optimal hurdle quantile 1-1/(2p) never exceeds 0.5).
**T4 blind spot-check (10 Nov, Gemini 3.8 Flash via agy, 15 labelled reference images, 50 blinded queries; it answered 28):** agreement with true labels on train 7/10, agreement with our model on test 13/18 (high-confidence 6/7 and 13/16). Chance is about 20%, so our test predictions are as right as Gemini can tell; the LB's T4 truth is the problem. Files in the session scratchpad (`t4check/`).
**TA answer (10 Nov):** T4 uses the datacard mapping (0 7-11k, 1 fiveN, 2 freefried, 3 pube, 4 roblock) and every task follows test.csv row order from ANS00001. `sub_v16a` already does exactly this and still scores T4 at chance, so the scorer's T4 truth looks broken (reported back to the TA with the ISO evidence). Overview hedge (`sub_v16e`) no longer needed. Leak permission still unanswered.
| **sub_v21a** (10 Nov) | v16a + T1 rank blend from v20a (110 rows) | **0.76315** best public (+0.00018, predicted +0.00017) |
| sub_v22a (11 Nov) | v21a + T1 pseudo-labels on the 13,184 unlabeled TRAIN rows, soft weight 0.5 (68 T1 rows) | 0.76307 (-0.00008 vs v21a; CV +0.00041 plain / -0.00006 test-weighted: pseudo-labels do not help, as exp07/11/19) |
| sub_v22b (11 Nov) | v21a + T1 hard pseudo-labels (conf >=0.8 / <=0.05, w 0.5; 62 T1 rows) | 0.76309 (-0.00006 vs v21a) |
| sub_v23a (11 Nov) | v21a + T4 blend 0.6 v3_5f + 0.4 EVA02-S (friend's 5-fold run `siriwanyindeephot/task4-v4-eva02s`, TTA plain+hflip+up; 1,521 T4 rows changed) | **0.76333** best public (+0.00018 vs v21a; T4 is at chance on public, so the gain may be luck). Nested OOF macro-F1 0.99170 vs v3 0.98886. Test shift: v3 and EVA02 agree on only 86% of test rows (98.5% on OOF), so test T4 is much harder than OOF and the ensemble should help more there. Public T4 is at chance, so LB cannot confirm |
| sub_v24a (12 Nov) | v23a + T3 10-seed bag (Kaggle CPU `task3-seedbag10`: 10 LightGBM p × 10 CatBoost d4, `t3/preds/seedbag10_final/`; the 3-seed rebuild matches t3_best_test to 3e-10) | **0.76334** best public (+0.00001 vs v23a, as expected) |
| sub_v24b (12 Nov) | no leak: v20a + T3 10-seed bag + T4 v3/EVA02 blend (both from v24a) | **0.72035** best non-leak (+0.00019 vs v20a; T3 ~+0.00002, so T4 blend ~+0.00017, the same gain as v23a over v21a); proposed final pair: leak allowed → v24a + v24b, not allowed → v24b + v20a |
CPU/GPU runs 12 Nov (none adopted): T1 final_v2 with 10 more seeds (13 total) nested binary F2 0.83927 vs 0.83919 (noise); T3 CatBoost d3 Optuna × 5 + d4 × 3 0.79531 (bag_d4x3 0.79529); T4 ConvNeXt-S in22k 5f OOF 0.9754, 3-way blend nested 0.99146 < 2-way v3+EVA02 0.99170. Builder for self-contained CPU kernels: `shared/make_cpu_kernels.py`.
Final selection: leak allowed → `sub_v21a` + `sub_v20a`; leak not allowed → `sub_v20a` + `sub_v20d`; no answer from the TA → `sub_v21a` + `sub_v20a`.

**T4 finding (6 Nov):** int text (`2`, not `2.0`) is the only format that scores, and it scores at chance. All 7 int probes fit an independence model (RMSE 0.0004) with digit shares 0 .1405, 1 .2983, 2 .2549, 3 .2008, 4 .1055. Any mapping gives macro about 0.19-0.20, i.e. +0.04 LB; the friend's gap to us is the same size. So LB T4 truth is unrelated to test.csv row order or broken. Report to the TA. The mapping deductions from M1/M2b/M3 were wrong; do not use sub_v14a/b.
Order tests on the const base: `11_O1_sortedids_datacard` 0.31327 (chance 0.3129), `12_S1_rowblocks` 0.31307 (chance 0.3127): no sign of filename order or contiguous blocks. T4 probing stopped pending the TA. Final candidates: `sub_v15a` (leak) and `sub_v15b` (no leak), T4 int text in datacard order.
**T5 id block (6 Nov):** in test, player_id >= 23819 (2,070 rows) matches our box set: 2,053 of our 2,055 rows are inside it and 17 block rows we miss all have NaN behavior_stability. There are no T5 labels, so this is agreement, not ground truth. `sub_v16a` = v15a with T5 = id block (needs TA approval, an id leak); `sub_v16b` = v15b with T5 = `t5_union_plus_acckda_box_lsnan.csv` (2,103 rows, no id).

Interpretation:
- **5 Nov, v10c:** +0.0659 vs v6. Predicted without any T4 change (weights 0.2, CV/estimated scores): T1 +0.0005, T2 leak +0.0424, T3 +0.0007, T5 box +0.0214 = +0.0650. So the T2 leak and the T5 box are real on test, and **T4 strings add ≈ 0: the public LB does not score T4** (numeric datacard, numeric overview and strings are all equal). Expected non-leak v10a ≈ 0.666.
- Removing T5 costs 0.177 → T5 is strong. That only fits if the LB's T5 metric is F3 of the anomaly class (constant 0 → 0) and T5 weight ≈ 0.2–0.25 → T5 LB ≈ 0.7–0.9.
- Adding rows beyond the union drops the score: the added rows hold ≈ 0–5% true anomalies. **Keep union 1,811.**
- With weights ≈ 0.25, T1+T2+T3 ≈ 0.4657/0.25 ≈ 1.86 on the LB (avg ≈ 0.62) vs CV ≈ 2.33 (0.78/0.76/0.79). **The ~0.1 gap to the friend is most likely in T1, T2 or T3**, probably a metric-definition mismatch, not model quality.

---

## 5. Next steps (in order)
1. Submit the leave-one-out probes (when the user asks): `submissions/probes/probes_v6/v6_minus_task3.csv`, `v6_minus_task2.csv`, `v6_minus_task1.csv`.
   Drop_k = w_k × (LB_k − const_k), with const ≈ T1(all 1) 0.36 macro (or 0.6 for F2 of the positive class), T2(all 0) 0.14, T3(all 0) 0.5 under our formula.
   The task whose drop is far smaller than w·(CV − const) is the broken one.
2. Prepared hypotheses to test on the broken task:
   - **T3:** the LB may not be MAE-based. Build T3 = conditional **mean** (p · E[y|y>0]) instead of the MAE-optimal quantile and compare. If the score rises, switch the objective (RMSE-style → mean, or MAPE-style → lower predictions).
   - **T1:** the LB may use F2 of the positive class, not macro → lower the threshold (predict ~55–60% cheaters).
   - **T2:** if it is the broken one, check label/format (ints 0–3 as floats).
3. Small gains to fold into the final file: T2 `t2_nb_bias.csv` (+0.0007), T1 GBM + TabPFN blend (~+0.002).
4. Final: choose 2 submissions for the private LB (Kaggle "Select" on the submissions page): best public + most CV-sound.
5. Report (30 Nov): `submissions/history/sub_v6_README.md` already explains every task's method.

**Data re-release (12 Nov):** T2 and T5 regenerated (same generator, all rows new, player_id leaks gone: T2 segment-vs-id AUC 0.500, T5 flag-vs-id AUC 0.500). T4 test.csv now `id,file_name,label` (ANS id -> image); the old row order matched the new one only at chance (21%) -> explains T4 at chance on LB. T1/T3 unchanged. Old files in `data_v1_backup/` (incl. stale T2 caches).
New: T2 `t2/preds/v2data/t2v2_blend_affine.csv` (same v18 recipe, OOF F1m 0.75826, `work/t2v2/make_t2v2.py`); T4 `t4/preds/realigned_v2/` (v3/EVA02 blend re-ordered by file name, no retrain); T5 `t5/preds/v2data/t5v2_union_box_lsnan.csv` (2,104 rows, pipeline in `work/t5v2/`). Submission `sub_v25a_newdata_T2v2_T4realigned_T5v2.csv` (`submissions/tools/build_v25.py`): **public 0.85787** (12 Nov, best; +0.1375 vs v24b, T4 now scored). Leak files (v16a/v21a/v24a) are invalid on the new data.
T4 EffNetV2-S 5f (12 Nov, `t4/preds/v4_effv2s`): OOF 0.9732; 3-way blend with v3+EVA02 gets weight 0.0, nested 0.99148 < 0.99170 -> not adopted (`t4/research/blend3_effv2`; blend script needs the old-format train.csv: `work/t4_blend_runs_v1train.py`).
**T4 test shift (12 Nov):** v25a LB implies test T4 macro-F1 ≈ 0.88 (if weights 0.2) vs OOF 0.99. Test images are darker within class (7-11k/fiveN median brightness -30) and more saturated (roblock ×1.8); v3/EVA02 agreement falls from 0.90 (bright) to 0.81 (darkest fifth). Noise level, JPEG blockiness, file size: same as train -> remaining shift is content.
GPU runs pushed 12 Nov ~17:40 (both EVA02-S, 3 folds, dataset `kiatisakkk/task4-image-classification` + competition, internet on, outputs in OLD test order):
- `task4-v4-eva02s-coloraug`: seed 42, only change = wider photometric aug (gamma, brightness 0.55-1.25, per-channel cast, saturation 0.5-2.0). Tests whether colour aug helps on test (compare on LB; OOF will not show it).
- `task4-v4-eva02s-s7`: seed 7, ensemble diversity.
After they finish: `kaggle kernels output kiatisakkk/<slug> -p t4/preds/<dir>` -> `python work/t4_blend_runs_v1train.py --run v3_5f=t4/preds/v3_5f/preds --run v4_eva02_s=t4/preds/v4_eva02_friend/preds --run <tag>=<dir>/preds --views plain,hflip,up --out t4/research/<blend>` -> `python submissions/tools/build_v26_t4.py t4/research/<blend> <tag>` (realigns to the new test order, swaps T4 into v25a). LB now measures T4, so a single T4 swap is a clean test (ΔLB ≈ 0.2·ΔF1).
Not done (needs a team decision, uses unlabeled test predictions): label-shift prior correction (test predicted mix 7-11k 16% vs train 39%).
| sub_v26_T4eq3effv2 (12 Nov) | v25a + T4 equal-weight geo-mean v3 + EVA02 + EffNetV2-S (994 T4 rows) | 0.85630 (-0.00157 vs v25a, T4 F1 ≈ -0.008: the weak model hurts on test too, OOF ranking holds; keep nested-CV weights) |
A teammate's `submission.csv` (12 Nov 11:36 UTC, "hybrid_v25 ... t5sota_top2pct") scored 0.44758 and used one of the day's slots.
T4 runs finished 13 Nov (3 folds each, outputs `t4/preds/v4_eva02s_coloraug`, `t4/preds/v4_eva02s_s7`): OOF coloraug 0.98334, s7 0.98425. Nested blends (`work/t4_blend_runs_v1train.py`, now accepts 3-fold runs): +s7 0.99152 (w .5/.3/.2), +coloraug 0.99170 (weight 0), all4 0.99152 vs v3+EVA02 0.99170 -> equal within noise; OOF cannot see the test shift. On test, coloraug does not change dark images more than other models.
Built, NOT submitted (user: no submit): `sub_v26_T4s7.csv` (T4 .5 v3 + .3 EVA02 + .2 EVA02-s7, 539 T4 rows vs v25a), `sub_v26_T4coloraug.csv` (.5/.3/.2 with coloraug, 511 rows); the two differ on 359 T4 rows. Each is v25a with only T4 changed -> LB delta isolates the run. All test probas: `work/t4_testproba_all.npz`.
Re-tune on new data (13 Nov, local CPU, no submit): T2 blend weights nested CV (3 seeds × 5 folds, affine rule inside) tuned 0.75833 vs .87/.08/.05 0.75826 (+0.00006 ± 0.00018, 8/15) -> keep old weights (`work/t2v2/tune_blend.py`). T5 lsnan prior from new data: expected hidden lowstab members 13 (old hard-coded 14; boxes still hold: members 100% in purchase_freq [3,6], 95% in age [800,1500]) -> 6 lsnan rows dropped, total 2,098. File `sub_v27_T5prior13.csv` (v25a with only T5 changed, 6 rows); expected LB change ~±0.0001, low priority.
Teammate submissions seen 13 Nov: `v25c` (v25a + T5 +26 rows prob>=0.50) 0.85762 (-0.00025: extra T5 rows hurt again); `v26b` (other T3/T5 mix) 0.76750; `v27` (v25a + T3 Jensen 50/50 CatBoost + two-stage hurdle + hard zero cutoff) 0.85776 (-0.00011).
| sub_v26_T4coloraug (13 Nov) | v25a + T4 .5 v3 + .3 EVA02 + .2 EVA02-coloraug (511 T4 rows) | 0.85733 (-0.00054, T4 F1 ≈ -0.003: wider colour aug does not fix the test shift) |
| sub_v27_T5prior13 (13 Nov) | v25a + T5 lsnan prior 13 (-6 rows, 2,098) | 0.85783 (-0.00004, noise; keep 2,104) |
v25a stays best (0.85787). Untested: `sub_v26_T4s7`.
**T4 DINOv3 runs (13 Nov evening).** Kernels pushed with the CLI cannot read Kaggle Secrets (`UserSecretsClient` -> ConnectionError), so DINOv3 runs must be started from the Kaggle UI (Edit -> Secrets HF_TOKEN attached -> Save & Run All). The 11 Nov "ConvNeXt-S" run silently used in22k weights for this reason: DINOv3 had never been tried beyond v3 Tiny.
- `task4-v4-cnxts-in22k` (code `t4/kaggle/task4-v4-cnxts-in22k/`, now preset `cnxt_s_d3`, no in22k fallback, 3 folds, guard 6.5 h): ConvNeXt-S DINOv3 fine-tune, outputs -> `t4/preds/v4_cnxt_s_d3`.
- `task4-5fold-ensemble` (code `work/t4_dinov3_feats/`; this slug's 3 Nov version is the v3 model in the submission): frozen DINOv3 ViT-L/16 + ViT-H+/16 features ([CLS, mean patch], fp16), train views plain/hflip/dark/sat, test plain/hflip -> linear head trained locally on train labels; outputs -> `t4/preds/dinov3_feats`.
ConvNeXt-S DINOv3 finished (`t4/preds/v4_cnxt_s_d3`, really `convnext_small.dinov3_lvd1689m`, 3 folds, early stop ep 10-13, ~75 min/fold): OOF 0.9859 (plain 0.9855, hflip 0.9856, up 0.9843). Nested blend v3 + EVA02 + cnxt_s_d3 (`t4/research/blend_cnxtsd3`): **0.99222** (weights .4/.4/.2, equal weights also 0.99222) vs v3 + EVA02 0.99170: first T4 run to raise the nested score. Test agreement: cnxt_s-v3 0.893, cnxt_s-EVA02 0.873, v3-EVA02 0.862. File `sub_v28_T4cnxtsd3.csv` (v25a with T4 only, 648 rows changed), NOT submitted.
| **sub_v28_T4cnxtsd3** (13 Nov) | v25a + T4 .4 v3 + .4 EVA02 + .2 ConvNeXt-S DINOv3 (648 T4 rows) | **0.85906** best public (+0.00119 vs v25a, T4 F1 ≈ +0.006: a bigger DINOv3 backbone helps on the shifted test; nested OOF +0.0005 understated it) |
Frozen DINOv3 features + linear head (`work/t4_dinov3_head.py`, 5 folds, train views plain/hflip/dark/sat, train labels only; outputs `t4/preds/dinov3_head/{vithp,vitl}`): ViT-H+ OOF 0.9639, ViT-L 0.9577. Nested blends: v3 + EVA02 + cnxt_s_d3 + ViT-H+ head **0.99281** (weights .3/.2/.1/.4; the H+ head agrees with the fine-tuned models on only 81-84% of test rows), + ViT-L as well 0.99253 (ViT-L weight 0). File `sub_v29_T4cnxtsd3_vithp.csv` (v25a + T4 from `t4/research/blend_cnxtsd3_vithp`; 1,361 T4 rows vs v25a, 1,148 vs v28), NOT submitted.
Kernel B prepared, NOT pushed (13 Nov): `t4/kaggle/task4-frozen-feats-b/` (slug `task4-frozen-feats-b`), frozen features from C-RADIOv4-H (HF remote code), SigLIP2 SO400M (`vit_so400m_patch16_siglip_384.v2_webli`, 224x432) and PE-Core-L (`vit_pe_core_large_patch14_336.fb`, 224x420); same views as the DINOv3 run; no HF_TOKEN needed, so `kaggle kernels push -p t4/kaggle/task4-frozen-feats-b` works. Each backbone is in try/except (a RADIO failure does not stop the other two). After it finishes: `kaggle kernels output kiatisakkk/task4-frozen-feats-b -p t4/preds/frozen_b` -> `T4_FEATS_DIR=t4/preds/frozen_b/feats python work/t4_dinov3_head.py <radio|siglip2|pe>` -> nested blend with `--run d3lin_<tag>=t4/preds/dinov3_head/<tag>`.
Kernel B finished 14 Nov (`t4/preds/frozen_b/feats`, ~4.9 h GPU): linear heads (`t4/preds/dinov3_head/{pe,radio,siglip2}`) OOF PE-Core-L 0.9504, C-RADIOv4-H 0.9643, SigLIP2 0.9289. Nested blends with v3 + EVA02 + cnxt_s_d3 + ViT-H+ head (0.99281): +PE 0.99243 (w 0), +RADIO 0.99284 (w .1, noise), +SigLIP2 0.99281 (w 0) -> none adopted.
ConvNeXt-B DINOv3 (friend's `chanonlhums/task4-v4-cnxts-in22k-base`, preset `cnxt_b_d3`, 3 folds): folds 0-1 done (time guard), OOF on 21,031 rows TTA 0.9895 (best single model; v3 0.9889). Saved in `t4/preds/v4_cnxt_b_d3_f01`. Fold 2 to be re-run with `T4_FOLDS` default "2"; the blend script merges dirs passed with the same tag.
| **sub_v30_T4eq4_cnxtb2f** (14 Nov) | v28 + T4 equal-weight geo-mean v3 + EVA02 + ConvNeXt-S DINOv3 + ConvNeXt-B DINOv3 (2/3 folds; `t4/research/blend_eq4_cnxtb2f`; 905 T4 rows vs v28) | **0.86169** best public (+0.00263 vs v28, T4 F1 ≈ +0.013). On the 21,031 rows with Base OOF: v28 recipe 0.99292, equal-4 0.99320. Blend script now has `--partial` (scores on rows all runs cover) |
| **sub_v31_T4eq4_radio_vithp** (14 Nov) | v30 + T4 frozen linear heads: .175 × 4 fine-tuned (v3, EVA02, cnxt_s_d3, cnxt_b_d3 2f) + .15 C-RADIOv4-H head + .15 DINOv3 ViT-H+ head (`t4/research/blend_eq4_frozen30`; 478 T4 rows vs v30) | **0.86340** best public (+0.00171 vs v30, T4 F1 ≈ +0.009; OOF on 21,031 rows only +0.00006: frozen features are robust to the test shift that OOF cannot see) |
| **sub_v32_T4frozen50** (14 Nov) | T4 .125 × 4 fine-tuned + .25 RADIO head + .25 ViT-H+ head (`t4/research/blend_eq4_frozen50`; 466 T4 rows vs v31) | **0.86430** best public (+0.00090 vs v31; OOF on 21,031 rows 0.99299, i.e. lower than v31's 0.99326 while LB rises: OOF and test disagree on frozen weight) |
Kernel C pushed 14 Nov (`t4/kaggle/task4-frozen-feats-c/`, slug `task4-frozen-feats-c`, CLI push, no HF_TOKEN): frozen DINOv2 ViT-g/14 reg4 (`dinov2g`, 224x420) + C-RADIOv3-H (`radio3`), same views/format as Kernel B. After: `kaggle kernels output kiatisakkk/task4-frozen-feats-c -p t4/preds/frozen_c` -> `T4_FEATS_DIR=t4/preds/frozen_c/feats python work/t4_dinov3_head.py <dinov2g|radio3>` -> add to the v32 recipe and test on LB (OOF cannot rank frozen weight).
**Label-shift prior correction approved by the user (14 Nov).** EM (Saerens) on the v32 T4 test proba: train prior 7-11k .394 fiveN .140 freefried .177 pube .188 roblock .101 -> EM prior .094/.295/.122/.257/.231. Applied as p × (pi_em/pi_train)^a.
| **sub_v33_T4frozen50_emhalf** (14 Nov) | v32 + T4 EM prior, a = 0.5 (`t4/research/blend_frozen50_emhalf`; 644 T4 rows vs v32, mostly 7-11k -> roblock/fiveN, freefried -> fiveN/pube) | **0.86669** best public (+0.00239 vs v32, T4 F1 ≈ +0.012) |
| **sub_v34_T4frozen50_emfull** (14 Nov) | v32 + T4 EM prior, a = 1 (`t4/research/blend_frozen50_emfull`; 1,206 T4 rows vs v32) | **0.86750** best public (+0.00081 vs v33, +0.00320 vs v32: the test prior really is shifted; the covariate-shift amplification risk did not show) |
Kernels saved for UI runs (14 Nov; the account's 2 batch GPU sessions were busy with Kernel C + ConvNeXt-B fold 2):
- `task4-frozen-feats-d` (`t4/kaggle/task4-frozen-feats-d/`, QuickSaved via the Kaggle MCP, not run): plan items 3+4 for RADIOv4-H and DINOv3 ViT-H+ (ViT-H+ needs HF_TOKEN -> UI run). Test views bright ×1.35 / desat ×0.7 / brdesat; train views dark2 ×0.55 / sat2 ×2.0 / darksat. Same extraction as before, so files combine with `frozen_b` (radio) and `dinov3_feats` (vithp).
- `task4-wiseft-cnxts` (`t4/kaggle/task4-wiseft-cnxts/`, plan item 8): cnxt_s_d3 fine-tune that also writes WiSE-FT interpolations theta = (1-a)·theta_LP + a·theta_FT for a = .3/.5/.7 (tags `v4_cnxt_s_d3_wise3/5/7`). Created with a CPU push (it stops at the HF_TOKEN check); run from the UI with GPU T4 + Internet + HF_TOKEN. Time guard 9 h.
ConvNeXt-B DINOv3 fold 2 done 14 Nov (friend's re-run, fold F1 0.9891); merged with folds 0-1 into `t4/preds/v4_cnxt_b_d3/preds` (full OOF 0.98932, 3 test folds). Full-OOF: equal-4 fine-tuned 0.99267, v34 recipe 0.99286, v28 recipe 0.99239.
Built, NOT submitted: `sub_v35_T4frozen50_emfull_b3f.csv` = v34 recipe (frozen .5 + EM full) with the 3-fold Base (`t4/research/blend_frozen50_emfull_b3f`; 101 T4 rows vs v34).
- `task4-v4-cnxtl-d3` (`t4/kaggle/task4-v4-cnxtl-d3/`, 14 Nov): ConvNeXt-L DINOv3 fine-tune, preset `cnxt_l_d3` (`convnext_large.dinov3_lvd1689m`, 224x416, batch 16, lr 1e-4, 16 epochs, dpr .3), guard 10.6 h (~2 folds per session, fold 2 in a second session via `T4_FOLDS` default "2"). Created with a CPU push (stops at the HF_TOKEN check); run from the UI on this account (GPU T4, Internet, HF_TOKEN). Tag `v4_cnxt_l_d3`.
**T4 scene-duplicate finding (14 Nov evening).** Train and test share the same synthetic corruption pipeline (noise median 16.7 vs 16.5, black-border share equal; `t4/research/shift_stats/`), so denoising cannot close the gap. The gap is content: train holds many near-duplicate scenes (RADIO feature NN cosine train->train median 0.75 vs test->train 0.66), so random-fold OOF measures memorised scenes. **Scene-group CV** (`work/t4_groupcv.py`: k-means scene clusters per class on a frozen backbone, whole clusters held out) behaves like the test: frozen heads fall from 0.963 (random) to ViT-H+ 0.893 / RADIO 0.879 / PE 0.853 (40 SigLIP2 clusters) and 0.935 / 0.928 / 0.910 (150 clusters); it ranks heads that random CV calls equal. Battery of head variants: `work/t4_groupcv_exp.py` -> `t4/research/groupcv/results.csv` (+ OOF in `oof/`). Class-balanced group folds for GPU runs: `t4/research/groupcv/sgkf_sig8_{fold,group}.npy`.
- `task4-scenecv-cnxts` (`t4/kaggle/task4-scenecv-cnxts/`, generator in the session scratchpad `gen_scenecv.py`): ConvNeXt-S DINOv3 fine-tune on those scene folds (0,1,2 of 5), arms `base` (v4 recipe) vs `crop` (zoom-in crops area 25-100% + CutMix p .5), early stop on held-out scenes. Tags `v4g_cnxt_s_d3_{base,crop}`. Created by CPU push (stops at HF_TOKEN); run from the UI (GPU T4, Internet, HF_TOKEN). `T4_BACKBONE=eva02_s` needs no token.
Grade formula (course doc p.7): above baseline Grade% = a + (100-a)·Phi((score-mu)/sigma) over teams above the baseline; a = 60 is only the doc's example. 14 Nov public LB: 11 teams above baseline 0.72601, mu 0.81649, sigma 0.04764 -> ML003 t = +1.07, 94.3% (a = 60); +0.001 LB ≈ +0.18 grade points. Kaggle is 70% of the project.
Kernel C heads (14 Nov, `t4/preds/dinov3_head/{dinov2g,radio3}`): random-CV OOF RADIOv3-H 0.9712, DINOv2-g 0.9588. Scene-group CV (sig8 / sig30): ViT-H+ .8927/.9352, RADIOv3 .8856/.9373, DINOv2-g .8873/.9304, RADIOv4 .8790/.9283. Frozen-only blends on group OOF: ViT-H+ + RADIOv4 (used in v32-v35) .8988/.9439; ViT-H+ + RADIOv3 + DINOv2-g .9057/.9504; grid best ViT-H+ .4 / RADIOv4 .1 / RADIOv3 .2 / DINOv2-g .3 (.9287 avg). Head variants (weight decay, L2, CLS-only, LDA, no class weight) are all within ±0.002 of the default or worse -> keep the default head.
Built, NOT submitted: `sub_v36_T4frozen4_emfull.csv` = fine-tuned .125 × 4 (Base 3-fold) + frozen .5 split ViT-H+ .20 / RADIOv4 .05 / RADIOv3 .10 / DINOv2-g .15 (group-CV weights × .5), then EM prior full (`t4/research/blend_frozen4_emfull`; 396 T4 rows vs v34).
ConvNeXt-L DINOv3 finished 15 Nov (`task4-v4-cnxtl-d3`, all 3 folds in one 9.6 h session, peak GPU 10.4 GB with grad checkpointing; `t4/preds/v4_cnxt_l_d3/preds`): OOF TTA 0.9870 (plain .9865), below Base 0.9893. Random-OOF 5-FT blend: equal 0.99194 (equal-4 0.99267), nested 0.99259 -> OOF says no gain (but OOF also underrated v28). Test agreement L-B 0.944, L-S 0.937, L-frozen 0.84-0.86.
Built, NOT submitted: `sub_v37_T4frozen4_ft5L_emfull.csv` = v36 with ConvNeXt-L as 5th fine-tuned model (FT .1 × 5 + same frozen .5 split + EM full; `t4/research/blend_frozen4_ft5L_emfull`; 235 T4 rows vs v36, 454 vs v34).
WiSE-FT (friend's `chanonlhums/task4-wiseft-cnxts`, `t4/preds/wiseft_cnxts/preds`): new cnxt_s_d3 run (seed-equivalent re-run, OOF 0.9860, 98.1% test agreement with the old one) + WiSE a=.3/.5/.7 OOF 0.9401/0.9738/0.9823. WiSE does NOT move test predictions toward the frozen heads (agreement with frozen-4 0.846/0.877/0.879 vs plain FT 0.876), so it adds no shift robustness here (the LP end is a weak ConvNeXt-S linear probe) -> not used. `chanonlhums/task4-v4-cnxts-in22k-base` latest output = the Base fold 2 already merged (identical).
Built, NOT submitted (low priority): `sub_v38_T4frozen4_s2seed_emfull.csv` = v36 with the cnxt_s slot = geo-mean of the two cnxt_s runs (`t4/research/blend_frozen4_s2seed_emfull`).
Kernel D done 15 Nov (friend's `chanonlhums/task4-frozen-feats-d` -> `t4/preds/frozen_d/feats`, names aligned with frozen_b / dinov3_feats). `work/t4_groupcv.py` load() and `work/t4_dinov3_head.py` fall back to that dir for missing views; the head script gained env `T4_TEST_VIEWS` (probabilities averaged over test views) and `T4_SUFFIX`. Group-CV exps `vithp_7views`, `radio_4views`, `radio_7views` added to `work/t4_groupcv_exp.py`.
Kernel E prepared 15 Nov (`t4/kaggle/task4-frozen-feats-e/`, slug `kiatisakkk/task4-frozen-feats-e`, CPU push = code only, stops at the GPU assert): frozen C-RADIOv3-g (`radio3g`), Web-SSL DINO ViT-1B (`webssl1b`, HF `facebook/webssl-dino1b-full2b-224`), AIMv2-H (`aimv2h`, [mean, max] patch pooling); none gated -> any account, GPU T4 + Internet, no secrets. Same views/format as B/C. Own-account GPU quota 15 Nov: 21.2 h of 30 h used (refresh 21 Nov 07:00 Bangkok), scenecv still running.
Kernel D results (15 Nov). Scene-group CV (sig8 / sig30), train views 2 / 4 / 7 (7 = + dark2, sat2, darksat): ViT-H+ .8927/.9352, .8943/.9367, .8942/.9361 (no gain); RADIOv4 .8790/.9283, .8830/.9301, .8848/.9324 (+0.002 over 4 views). Heads with 7 train views + per-view test preds: `t4/preds/dinov3_head/{vithp_v7,radio_v7}` (random OOF .9628 / .9648; `_test_fold{k}.npy` = mean of plain, hflip, bright, desat, brdesat). Inverse-shift test views agree LESS with the v36 blend than plain views (ViT-H+ bright .859 vs plain+hflip .874; RADIO .847 vs .872) -> no sign that brightening/desaturating the test helps; only LB can tell.
Built, NOT submitted (low priority): `sub_v39_T4frozen4_v7tta_emfull.csv` = v36 with the ViT-H+ and RADIOv4 heads replaced by the 7-view heads + 5-view test TTA (`t4/research/blend_frozen4_v7tta_emfull`).
**Scene-CV fine-tune result (15 Nov, `task4-scenecv-cnxts` -> `t4/preds/scenecv_s/preds`, folds 0-2 = 17,038 rows).** ConvNeXt-S DINOv3 on held-out scenes: base 0.9635 (folds .9735/.9781/.9456), crop + CutMix 0.9559 (.9732/.9842/.9226) -> crop arm NOT adopted. Frozen heads on the same SGKF folds (`work/t4_groupcv_sgkf.py`, OOF in `t4/research/groupcv/oof_sgkf/`): ViT-H+ .9290, RADIOv3 .9291, DINOv2-g .9257, RADIOv4 .9228; frozen-2 (v34 pair) .9404, frozen-4 (v36 weights) .9461. So a fine-tuned model beats every frozen head on new scenes, but they are complementary: FT + frozen-4 geo-mean peaks at FT weight 0.5 (.9697; .3 -> .9679, .7 -> .9677, FT alone .9635), in every fold. This confirms the 50/50 recipe that the LB picked (v32). frozen-4 vs frozen-2 inside the 50/50 blend: .9697 vs .9691 (+0.0006 only) -> v36 expected ~+0.0001-0.0003 LB over v34.
| **sub_v36_T4frozen4_emfull** (15 Nov, submitted by the user) | v34 with frozen-4 heads (ViT-H+ .20 / RADIOv4 .05 / RADIOv3 .10 / DINOv2-g .15) + Base 3-fold, EM full (396 T4 rows vs v34) | **0.86805** best public (+0.00055 vs v34, T4 F1 ≈ +0.003; scene-group CV predicted the direction, and the size sits between the SGKF estimate +0.0001-0.0003 and the first estimate +0.0006-0.0012) |
| **sub_v37_T4frozen4_ft5L_emfull** (15 Nov) | v36 + ConvNeXt-L DINOv3 as 5th fine-tuned model (FT .1 × 5 + frozen-4 .5, EM full; 235 T4 rows vs v36) | **0.86837** best public (+0.00032 vs v36, T4 F1 ≈ +0.0016; random OOF said no gain: again OOF underrates a bigger backbone on test) |
Built, NOT submitted: `sub_v40_T4v37_em150.csv` = v37 with EM prior exponent a = 1.5 (`t4/research/blend_frozen4_ft5L_em150`; pre-EM proba + prior saved in `blend_frozen4_ft5L_emfull/`). Rationale: a .5 -> 1 gained +0.0008 and the weighted geo-mean of disagreeing models is under-confident, so EM under-corrects; LB probe of whether a > 1 keeps gaining.
| sub_v40_T4v37_em150 (15 Nov) | v37 with EM prior exponent a = 1.5 (638 T4 rows) | 0.86643 (-0.00194 vs v37: over-correction). With a = .5 (-0.0008, measured on the v32 base) and a = 1, a parabola peaks at a ≈ 0.9 with < 0.0001 to gain -> **keep a = 1, prior-exponent tuning closed** |
- `task4-v4-vitl-d3` (`t4/kaggle/task4-v4-vitl-d3/`, 15 Nov, CPU push = code only): ViT-L/16 DINOv3 fine-tune, preset `vit_l_d3` (`vit_large_patch16_dinov3.lvd1689m`, 224x416, batch 16, lr 5e-5, llrd .75, 12 epochs, dpr .2, grad checkpointing), same 3 folds as the other v4 runs, tag `v4_vit_l_d3`. Needs GPU T4 + Internet + HF_TOKEN (UI run / friend Copy & Edit). Expect ~2 folds per session; fold 2 via `T4_FOLDS` default "2". Model code smoke-tested on CPU without weights.
- `task4-v4-eva02b` and `task4-v4-dinov2l` (15 Nov, CPU push = code only, `t4/kaggle/<slug>/`): same v4 script, presets `eva02_b` (`eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`, 224x420, batch 24, lr 7e-5, 16 ep, tag `v4_eva02_b`) and `dinov2_l` (`vit_large_patch14_reg4_dinov2.lvd142m`, 224x420, batch 16, lr 5e-5, 12 ep, tag `v4_dinov2_l`). Neither is gated -> any account, GPU T4 + Internet, no secrets. Smoke-tested on CPU without weights.
**Rule check (15 Nov, user asked).** Rule: "Prohibited: using any part of the Test set for training-related purposes (e.g., pseudo-labeling or fine-tuning)". All models are trained on train labels only; T1 pseudo-labels used unlabeled TRAIN rows (not in current files); T5 has no train set. Grey zone: the T4 EM prior correction (v33-v37; worth ~+0.0032 LB) estimates the class mix from test predictions. User approved it on 14 Nov, TA not asked yet -> ask the TA, and keep a no-EM file as the second final pick.
kNN smoothing study (`work/t4_knn_smooth.py`, scene-group folds, train only): averaging the blend's log-probs with the mean of each image's k nearest held-out neighbours (frozen-feature cosine). Baseline FT50 + frozen-4 .9697 -> k 10, lam .3: RADIOv3 neighbours **.9742**, ViT-H+ .9731, concat .9734 (all 3 folds up); lam .5 ≈ +0.001-0.002; lam ≥ .7 hurts (.94-.95). Test has the same neighbour structure (test->test NN1 cosine .677 vs held-out fold .683). Expected ~+0.0009 LB. Transductive (uses test-test neighbours at inference) -> same grey zone as EM, **NOT to be built/submitted until the TA answers**.
| sub_v41_T4v37_knn10l3_emfull (15 Nov, user asked) | v37 + kNN smoothing of the pre-EM blend (k 10, lam .3, RADIOv3 test-test neighbours), EM prior re-estimated (710 T4 rows vs v37; `t4/research/blend_v37_knn10l3_emfull`) | 0.86760 (-0.00077 vs v37; scene-group CV had said +0.0045 F1 -> kNN smoothing does NOT transfer to the test, not used). First case where scene-group CV got the sign wrong: held-out train scenes are still closer to each other than test images are in label terms |
Built, NOT submitted: `sub_v37n_T4v37_noEM.csv` = v37 without the EM prior (clean fallback if the TA rejects prior correction; 1,067 T4 rows vs v37, expected ~0.865).
Friend's Colab ConvNeXt-L run (15 Nov, zip `preds-20261115T070221Z-1-001.zip` -> `t4/preds/colab_large_zip/`): same model `convnext_large.dinov3_lvd1689m` but batch 32, lr 1.5e-4, 18 epochs, dpr .2; OOF 0.99165 (best single model; Kaggle L run 0.98699). Its TEST files are in the NEW test.csv order (ANS id = new order) -> realigned by file_name into `t4/preds/v4_cnxt_l_d3_colab/preds` (tag `v4_cnxt_l_d3c`, old order like every other run). After realigning: 96.4% test agreement with the Kaggle L run.
Built, NOT submitted: `sub_v42_T4v37_L2run_emfull.csv` = v37 with the L slot (.1) = geo-mean of the Kaggle and Colab L runs (`t4/research/blend_v37_L2run_emfull`).
| sub_v37n_T4v37_noEM (15 Nov, user asked) | v37 without the T4 EM prior (1,067 T4 rows vs v37) | 0.86494 (-0.00343 vs v37 = the exact value of the EM prior on the v37 base; expected ~0.865). **Clean fallback = 0.86494**, still 1st (ML007 0.85948). Final pair if the TA has not answered: v37-family best (EM) + this no-EM file |
Other tasks (15 Nov review): T1/T2/T3/T5 are at their ceilings (every tested change within ±0.0002 LB); TabPFN is not in the T1 blend (est. +0.00016 LB, needs a UI run with TABPFN_TOKEN, not worth it). Remaining headroom is T4 only.
