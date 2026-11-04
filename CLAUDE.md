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
python -m kaggle kernels output kiatisakkk/<slug> -p t123\preds_final\<dir>     :: download outputs + log
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
cpe342-kaggle-main/          competition data (01_binary, 02_multiclass, 03_regression, 04_image, 05_anomaly, sample_submission.csv)
t123/                        T1/T2/T3 code (task1_gbm_pipeline.py, task1_tabpfn.py, task3_hurdle_ziln_v2..v4.py, task3_nn_ziln.py)
t123/preds_final/            FINAL test predictions used in submissions
    t2_nb.csv                T2 Naive Bayes (used in v6)
    t2_nb_bias.csv           T2 NB + macro-F1 class-bias (+0.0007 CV, not yet submitted)
    v4/preds/                T3 v4 outputs (D_q, D_qw0.3, D_qw0.6 …)
    t1/t1_pipeline/          T1 GBM blend outputs
    t4/preds/                T4 5-fold outputs (task4_v3_5f.csv, test_proba.npy)
t123/t2_nb/                  T2 NB scripts (nb.py, pf.py = per-feature Optuna, best_nb_perfeature.json)
t123/t5/                     T5 scripts (t5_build_union.py, task5_tmae.py)
task4/                       T4 notebooks / task4_5fold.py
masking/                     T5 detector experiments + per-task probe plan
submissions/                 every submitted / prepared file + README per version
    probes_v6/               leave-one-task-out probes (see §4)
    probes_t5/               T5 sets (t5_groups.csv, 827/1034/1811/1919/2167/2800/3602)
```
The T3 NN test file is `NN_K3` (it arrived as an attachment; the blend is already inside `sub_v5/v6`).

---

## 3. What each task does now (= `sub_v6_t1gbmblend_t2nb_t3blend_t4ens5f_ovr_t5union.csv`)

### T1 — cheater detection, binary, **F2 macro**
- EDA: 16 pure-noise features. 15 signal features. Class-exclusive value ranges plus an overlapping core of ~55% (AUC ≈ 0.79 inside the core) → ceiling ≈ 0.78.
- Hidden identity `level_progression_speed = level · r / account_age_days`, r ~ U(0.15, 1.08) → features `age_lo, age_hi, age_filled, lv_lo, lv_hi`.
- Model: XGB + LightGBM + CatBoost, each tuned with Optuna (200 trials, 3-fold AUC), refit 5-fold. Rank blend weights xgb 0.18 / lgb 0.48 / cat 0.33. Threshold chosen for max F2 macro (predicts 40.4% cheaters).
- CV: AUC 0.8925, **F2m 0.7804**. Alternatives: HGB 0.7807, TabPFN 0.7799, TabPFN+HGB 0.7825, EBM 0.7786.
- Labeled rows only (~13k train rows have no label and are unused).

### T2 — player segment, 4 classes, **F1 macro**
- EDA: features are conditionally independent within class → Naive Bayes is Bayes-optimal. Latent 2×2 structure (engagement × spending). 16 noise columns removed. Informative categoricals: loyalty_tier, engagement_level.
- Model: histogram NB (quantile bins + Gaussian smoothing per continuous feature, Laplace α). Per-feature (bins, smoothing) tuned by Optuna (`pf.py`). NaN is marginalised (skipped).
- CV **F1m ≈ 0.755–0.758** (HGB 0.750, XGB 0.7498).
- Tested 4 Nov: EM prior estimation (Saerens) shows the test prior ≈ the train prior, so there is no shift. Class-bias tuning gives +0.0007 nested CV (`t2_nb_bias.csv`).

### T3 — spending, regression, **NMAE score** (we assume `1/(1+MAE/mean|y|)`, all-zero = 0.5; NOT verified on LB)
- ~48% zeros. Whales (segment 3) cause ~80% of the error.
- Exact fills: `historical_spending = total_transactions × avg_transaction_value`, `purchases_on_discount = total_transactions × discount_rate_used` (NaN 5.8% → 0.4%). Segment imputer (holdout acc 0.929).
- Hurdle: classifier p = P(y>0) + XGB multi-quantile; predict the conditional quantile q = 1 − 1/(2p) (0 if p < 0.5), which is MAE-optimal; (shift, κ) calibration on OOF.
- Blend (OOF weights): v4 D_q 0.38 + D_qw0.3 0.24 + D_qw0.6 0.06 + NN_K3 0.32 → **OOF 0.7925** (HGB two-stage 0.7843).

### T4 — game screenshots, 5 classes, **F1 macro**
- ConvNeXt-Tiny `convnext_tiny.dinov3_lvd1689m`, LP-FT, LLRD 0.8, EMA, 14 epochs, 5 folds, averaged softmax. OOF F1 0.9889.
- **The public LB ignores T4**: datacard order vs Overview order vs (friend) string labels all give an identical score. v6 keeps numeric Overview order (0 PUBE, 1 FreeFried, 2 FiveN, 3 Roblock, 4 7-11k) in case the private LB scores it. Datacard order = alphabetical (0 7-11k, 1 fiveN, 2 freefried, 3 pube, 4 roblock). `t4/preds/task4_v3_5f.csv` has both `pred_num` (datacard) and `pred_str`.

### T5 — account anomaly, unsupervised, **F3**
- `t5_build_union.py` reproduces the set exactly (1,811 rows):
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

Interpretation:
- Removing T5 costs 0.177 → T5 is strong. That only fits if the LB's T5 metric is F3 of the anomaly class (constant 0 → 0) and T5 weight ≈ 0.2–0.25 → T5 LB ≈ 0.7–0.9.
- Adding rows beyond the union drops the score: the added rows hold ≈ 0–5% true anomalies. **Keep union 1,811.**
- With weights ≈ 0.25, T1+T2+T3 ≈ 0.4657/0.25 ≈ 1.86 on the LB (avg ≈ 0.62) vs CV ≈ 2.33 (0.78/0.76/0.79). **The ~0.1 gap to the friend is most likely in T1, T2 or T3**, probably a metric-definition mismatch, not model quality.

---

## 5. Next steps (in order)
1. Submit the leave-one-out probes (when the user asks): `probes_v6/v6_minus_task3.csv`, `v6_minus_task2.csv`, `v6_minus_task1.csv`.
   Drop_k = w_k × (LB_k − const_k), with const ≈ T1(all 1) 0.36 macro (or 0.6 for F2 of the positive class), T2(all 0) 0.14, T3(all 0) 0.5 under our formula.
   The task whose drop is far smaller than w·(CV − const) is the broken one.
2. Prepared hypotheses to test on the broken task:
   - **T3:** the LB may not be MAE-based. Build T3 = conditional **mean** (p · E[y|y>0]) instead of the MAE-optimal quantile and compare. If the score rises, switch the objective (RMSE-style → mean, or MAPE-style → lower predictions).
   - **T1:** the LB may use F2 of the positive class, not macro → lower the threshold (predict ~55–60% cheaters).
   - **T2:** if it is the broken one, check label/format (ints 0–3 as floats).
3. Small gains to fold into the final file: T2 `t2_nb_bias.csv` (+0.0007), T1 GBM + TabPFN blend (~+0.002).
4. Final: choose 2 submissions for the private LB (Kaggle "Select" on the submissions page): best public + most CV-sound.
5. Report (30 Nov): `submissions/sub_v6_README.md` already explains every task's method.
