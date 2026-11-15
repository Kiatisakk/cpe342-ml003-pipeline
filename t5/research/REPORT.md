# T5 research: finding the missing anomalies

Date: 4 Nov 2026 (rounds 1 and 2). Data: `05_anomaly_detection/test.csv` only. **Nothing was submitted to Kaggle.**

## TL;DR

* **The generator is now reverse-engineered.** Every injected type replaces some metrics with **independent uniform draws per period, inside a fixed box with round bounds** (period-4 shocks for the three rule subtypes). There are 8 types:

  | type | rows | definition |
  |---|---|---|
  | takeover | 312 | p4: currency_transfers and location_changes replaced, play_consistency ×0.3–0.6 |
  | selling | 310 | p4: friend_removes and skill_trajectory replaced, skill/kda/accuracy ×1.5–3, reaction ×0.5–0.8 |
  | VPN | 204 (+1 geo-only) | p4: location_changes and purchase_frequency replaced, purchase_amount ×2–6.7 |
  | bot | 207 | ip_entropy [0, .2], kda_std [.05, .15], login_time_var [.1, .5], perf_stab and play_consistency [.95, .99], session_std [1, 5], device_count = 1 |
  | lowact | 266 | activity [.1, .3], purchase_amount [300, 500], skill_rating [2200, 2500] |
  | lowcomm | 237 | communication [0, .1], friend_adds [15, 25], guild_changes [2, 3] |
  | lowstab | 274 (+~14 hidden) | behavior_stability [.2, .4], account_age [800, 1500], purchase_frequency [3, 6], login_count ≈ ×0.5 |
  | **acckda (NEW)** | **244** | **performance_stability [.85, .95], kda_std [.70, 1.00], accuracy_std [9, 12]** |

* **The new type, found exactly.** 244 rows have all observed periods inside its box, with **0 inside the union** and **0.85 expected by chance** (≈ 99.7% precision). Narrowing any bound by 10% loses about 75 rows; widening it adds 1.
* **Accounting matches the LB.** K = 1,811 + 244 + ~14 lowstab rows hidden by NaN `behavior_stability` = **2,069**. That predicts union-only F3 = **0.8864**, against the LB-implied ≈ 0.886 ✔.
* **Union precision is about 100%.** The box rules reproduce bot, lowact and lowcomm row-for-row, and lowstab includes all 274.
* **Nothing else is left.** A box-pair search (1M pairs), level-adjusted variability scans, 20×20 pair grids over 263 per-period features, geo hotspots, and joint shock-signature checks are all at null level.
* **Recommended probe:** `t5_candidates/t5_union_plus_acckda_box.csv` (2,055 rows). Expected T5 F3 is **0.993** (from 0.886), worth about **+0.021** on the public LB (≈ 0.664).
* **Pipeline** (`t5_pipeline.py`, ~5 s, unsupervised) reproduces the union exactly and returns the same 244-row box by itself. The box is estimated from the EM cluster's confident members, with no hand-set bounds.

## Files (run from the repo root, in this order)

| file | what |
|---|---|
| `t5_discovery.py` | round-1 analyses (independence, signatures, tail co-occurrence, grid scans, cluster size); log in `out/discovery_log.txt` |
| `t5_pipeline.py` | unsupervised pipeline; writes `out/t5_pipeline_scores.csv` and `out/t5_pipeline_summary.json`; log in `out/pipeline_log.txt` |
| `t5_cluster_refine.py` | EM variants M3/M4/M6 and held-out-feature calibration; log in `out/cluster_refine_log.txt` |
| `t5_box_analysis.py` | round-2 box analyses and lowstab-NaN scores (`out/lowstab_nan_scores.csv`); log in `out/box_analysis_log.txt` |
| `t5_make_candidates.py` | writes `t5_candidates/*.csv` (`id` = ANS00001.. in test.csv row order, `pred` 0/1, LF) |
| `t5_swap_into_submission.py` | puts a candidate into the task5 column of a local full submission (e.g. sub_v6). It does not submit. |

```bash
python t5/research/t5_discovery.py
python t5/research/t5_pipeline.py
python t5/research/t5_cluster_refine.py
python t5/research/t5_box_analysis.py
python t5/research/t5_make_candidates.py
python t5/research/t5_swap_into_submission.py submissions/<sub_v6 file>.csv t5/research/candidates/t5_union_plus_acckda_box.csv submissions/sub_v7_t5box.csv
```

## Candidate table (ranked)

Expected F3 uses K = 2,069: the union is taken as 100% precise, with 244 box rows (1 of them chance) and ~14 hidden lowstab rows.

| rank | file | flagged | added | expected T5 F3 | confidence |
|---|---|---|---|---|---|
| ref | `t5_union_1811.csv` | 1,811 | 0 | 0.886 (LB ≈ 0.886 ✔) | = sub_v6 |
| **1** | **`t5_union_plus_acckda_box.csv`** | **2,055** | **244** | **0.993** | **very high**: exact box, chance ≈ 1 row |
| 2 | `t5_union_plus_acckda_box_lsnan.csv` | 2,103 | 292 | 0.997 (if calibrated) | box: very high; +48 lowstab-NaN rows: low (~11 true expected, 24%) |
| – | `t5_union_plus_acckda_p10/p50/p90.csv` | 2,086 / 2,045 / 1,988 | | 0.990 / 0.984 / 0.964 | superseded: Gaussian-EM cuts; p50 misses 22 box rows and adds 12 non-box rows |

## Probe order (only when the user asks to submit)

1. `sub_v6` with T5 = **box** (`t5_swap_into_submission.py`). Expected public ≈ 0.6429 + 0.2 × (0.993 − 0.886) ≈ **0.664**.
2. Optional: **box_lsnan**. Expected an extra ≈ +0.0006 on the LB, which is within public-subset noise. Keep it only if it is not lower.
3. If (1) does not rise, there is something wrong with the T5 metric assumption, not the set. Probe `p90` (177 rows, ~99% precise) to check.

---

## Round 1 (first session) — how the type was found

Notation: "means" = per-row mean of the 4 periods; robust z = (x − median)/(1.4826·MAD) on rows outside the union.

1. **Normal rows have independent features.** Max |corr| between the means of two different features is **0.032**. So any joint structure outside the union is injected.
2. **Known-group signatures** (median robust z):
   * bot: ip_entropy −5.9, login_time_var −2.1, session_std −2.0, kda_std −1.8, perf_stab +1.7, play_consistency +2.0
   * lowact: activity −2.1, purchase_amount +0.8, skill +1.1
   * lowcomm: comm −2.1, friend_adds +1.5
   * lowstab: behavior_stability −5.7, login_count −0.9, purchase_freq +1.1

   The 5 groups are mutually exclusive (827+207+266+237+274 = 1,811).
3. **Stationary groups are complete.** No outside rows carry the other signature features.
4. **Shock group is complete.** Every rule row has a p4 shock > 10 robust-z (or a geo jump). Outside, only a smooth skill_trajectory tail remains.
5. **Pairwise 5%-tail co-occurrence:** accuracy_std high × kda_std high = 222 observed vs 60 expected (**z = 20.9**). Every other pair is z ≤ 3.7.
6. **Triple grid scan (2,600 triples):** (accuracy_std, kda_std, performance_stability) **z = 19.8**, against a median of 3.2.
7. **Dynamics** (period std, Δ12, Δ34, slope) are at null level. **Discrete columns and flags** are at null level; the flags are enriched only in the rule group.
8. **Size:** the Mahalanobis column-shuffle-null excess plateaus at ≈ 230; EM n_eff = 237. **Isolation Forest** scores AUC 1.00 against the shocks and is blind to the dependence-only cluster.

## Round 2 (this session)

### R2.1 More scans for other types (all null)
* **Geo:** home lat/lon is uniform. The densest 0.5° cell holds 5 rows; the max period-to-period location spread is 2.1°.
* **Real vs column-shuffled classifier** (HistGB, 37 features): AUC 0.49. It is too global to see a 1% cluster. That is not evidence either way, so it is not used.
* **20×20 rank grids for every pair** of 263 features (means, raw per-period values, deviation of each period from the others; 22,789 cross-metric pairs): top z = 5.5, the expected maximum under the null.
* **Level-adjusted variability** (log std over periods minus a cubic fit on the row mean) × means, 5% tails: top z = 4.0. Null.
* **skill_trajectory p4 tail:** outside the union, the p4 tail is the *thinnest* of the 4 periods (period noise shrinks p1→p4 for everyone), and no deviation exceeds 20. Not a hidden selling group.

### R2.2 Sharper cluster: variability is a signature too
Members' performance_stability varies across periods about half as much as normal rows with the same level (period-sd 0.025 vs 0.044). EM with that feature (M4) halves the ambiguous rows (139 → 96 in 0.1–0.9).

The held-out calibration (`t5_cluster_refine.py`) uses features a model did not see. It shows the Gaussian posterior is **under-confident at the top** (band 0.5–0.9 is ≈ 100% true) and over-confident in 0.1–0.5. So the cluster has a hard edge. That led to the box.

behavior_stability is **not** part of the type (core and normal means are equal), so the M6 variant is rejected.

### R2.3 The box (`t5_box_analysis.py` §1–2)
* Members' **per-period** values are exactly inside perf_stab **[0.8502, 0.9498]**, kda_std **[0.7001, 1.0000]** and accuracy_std **[9.0008, 11.9975]**. Normal values pass these bounds on both sides.
* Uniform U(0.85, 0.95) has sd 0.029, which matches the low period noise measured in R2.2.
* **Rows with all observed periods inside: 244**, with 0 in the union. Narrowing a bound by 10% gives 152–171 rows; widening by 10% gives 245.
* **Null with within-metric period correlation.** Take, per metric, the distribution in normal rows of how many periods fall outside the box, and convolve over the 3 metrics:

  | out-of-box period values | observed | chance |
  |---|---|---|
  | 0 | 244 | 0.85 |
  | 1 | 8 | 7.1 |
  | 2 | 32 | 29.8 |
  | 3 | 85 | 90.5 |

  So precision is ≈ 99.7%, and there are **no** near-misses to recover.
* The pipeline's box comes from the posterior > 0.9 members (0.5/99.5% quantiles ± 2%): [8.95, 12.04], [0.696, 1.005], [0.849, 0.951]. It selects **the same 244 rows**.

### R2.4 The known types are boxes too (§1), so the union is ≈ 100% precise
The box rules reproduce **bot 207/207, lowact 266/266, lowcomm 237/237** with zero extra rows. All 274 lowstab members are inside their box.

### R2.5 Box-pair search for any remaining type (§3)
For every metric, take a grid of 13 points and use every interval up to half the range, giving 1,443 indicators "all 4 periods in [a, b]". Then test co-occurrence of every cross-metric pair against independence.
* **Calibration:** with all rows, bot appears at z = 126 (207 observed vs 1.7 expected).
* **Outside union + box:** top z = **4.5** over 1.04M pairs. That is null, so no other multi-metric type exists anywhere in the value range.

### R2.6 Shock subtypes (§4)
The rule group splits into takeover 312, selling 310, VPN 204 and 1 geo-only row. Some metrics are replaced (corr with the base ≈ 0) and others are multiplicative.

Each member shows 2–6 signature shocks beyond the members' 5% level. **No row outside the union shows even 2.** So the shock types are complete.

### R2.7 Lowstab members hidden by NaN (§5)
The union rule `behavior_stability < 0.55` is a single static column and is NaN for 5% of rows, so about 14 members are invisible to it.
* Among NaN-stability rows, 137 fall in the purchase_freq/age box against 113 by chance (excess 24 ± 11).
* A likelihood ratio on login_count (members ≈ ×0.5) and purchase_freq period-sd (uniform, so noisier) cannot separate them well. Posterior > 0.1 gives 48 rows with ~11 expected true.
* That is above the ~10% F3 break-even, but the confidence is low. It is offered only as candidate 2.

## Pipeline for the course report

`t5_pipeline.py` is unsupervised and needs no labels or hand-set thresholds for the new type:

| stage | model | finds | rows |
|---|---|---|---|
| A | robust-z (median/MAD) temporal-change detector p3/4 vs p1/2, + geo jump | takeover / selling / VPN | 827 |
| B0 | 1-D density-gap detector on 4-period means | lowstab, lowact, lowcomm, bot | 984 |
| B | dependency discovery: tail co-occurrence vs independence, graph components | feature subset of a hidden cluster | – |
| C | EM mixture: Gaussian cluster vs product-of-marginals background, with signature expansion (means, then level-adjusted variability) | posterior score for the new cluster | – |
| C2 | support estimation: per-period box from confident members, flag rows fully inside | new acckda type | 244 |
| D | Isolation Forest (soft score, check) | AUC 1.00 vs shocks, 0.885 vs final | – |
| vote | hard OR of A, B0, C2 | | **2,055** |

Overlap with the union: **1,811/1,811 reproduced, 0 extra**, plus 244 new rows = `t5_union_plus_acckda_box.csv` exactly.

## Dead ends (both rounds)
* Loosening stationary thresholds (§R1.3, and box rules show nothing outside).
* Single-period shocks in periods 1–3; skill_trajectory tail (R1.4, R2.1).
* Period-dynamics and variability clusters (R1.7, R2.1).
* Discrete/flag combinations (R1.7).
* Geo hotspots (R2.1).
* Global real-vs-shuffled classifier: too weak (R2.1).
* Isolation Forest / TMLP / TMAE extensions: unstructured tails.
* Gaussian-EM cuts p10/p50/p90 are superseded by the exact box.

## Round 3 — structure inside the rows still called normal (`t5_normal_structure.py`, `out/normal_structure_log.txt`)
* **4-feature scan:** all 14,950 combinations of 4 of the 28 means/statics, 5⁴ rank cells, on the 23,834 normal rows. Real top z = **6.27** (activity × friend_adds × reaction_time × account_age; every pair inside it is z ≤ 2.4). Six independent column-shuffle nulls give 5.48–6.62, so the real top is **inside the null range**. Calibration: putting the 244 new-type rows back gives z = **9.9**, so a type of that size would be found.
* **GMM (BIC) and HDBSCAN on all 28 dims:** real and shuffled data behave the same.
  * HDBSCAN's only extra cluster (3,877 rows) is the friend_removes = 0 clipping spike.
  * Neither method isolates the 244 new-type rows in calibration (they come out at 1% share), so full-dimensional clustering is too weak here.
* **t-SNE** (`figs/fig3_tsne.png`, 56 features):
  * bot, takeover, selling and VPN are separate islands.
  * The new type shows as a dense patch at the blob edge.
  * lowact, lowcomm and lowstab dissolve into the blob.

  So a smooth normal blob is **not** proof that no clusters remain. The low-dimensional scans are, and all of them are at null level.
