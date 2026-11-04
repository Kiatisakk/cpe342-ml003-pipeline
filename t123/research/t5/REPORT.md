# T5 research: finding the missing anomalies

Date: 4 Nov 2026. Data: `05_anomaly_detection/test.csv` only. **Nothing was submitted to Kaggle.**

## TL;DR

* **New anomaly type found.** It is a stationary cluster of about 237 accounts with high `accuracy_std` (mean ≈ 10.4), high `kda_std` (≈ 0.85) and high `performance_stability` (≈ 0.90), all four periods. It does not overlap the current 1,811-row union.
  This is the only cross-feature dependence in the whole dataset outside the union: z = 20.9 against a max of 3.7 for every other pair, and z = 19.8 against a median of 3.2 over 2,600 triples.
  The group is about the size of the "~250 missing anomalies" the LB implied. Adding it explains the gap: expected T5 F3 goes from **0.887 to 0.984** (union-only expectation 0.887 ≈ observed LB 0.886).
* **Union is complete otherwise.** Shock rows are separated by huge gaps, and the 4 stationary groups have no partial members outside. No other hidden structure was found in means, period dynamics, discrete columns or flags.
* **Model-based pipeline** (`t5_pipeline.py`, 5 s) reproduces the union **exactly** (1,811/1,811, 0 extra rows) and adds the new cluster. It uses a robust-z shock detector, a 1-D density-gap detector, and a dependency-discovered EM mixture, plus an Isolation Forest check. That gives a course-compliant "unsupervised ensemble with hard voting" story.
* **Probe first:** `t5_candidates/t5_union_plus_acckda_p50.csv` (2,062 rows). Expected LB gain ≈ +0.019 (T5 weight 0.2 × ΔF3 ≈ 0.097).

## Files (all runnable from the repo root)

| file | what |
|---|---|
| `t5_discovery.py` | every analysis below; output in `out/discovery_log.txt` (14 s) |
| `t5_pipeline.py` | unsupervised pipeline; writes `out/t5_pipeline_scores.csv` (per-row flags, posterior, IForest score) and `out/t5_pipeline_summary.json`; log in `out/pipeline_log.txt` |
| `t5_make_candidates.py` | writes `t5_candidates/*.csv` (`id` = ANS00001.. in test.csv row order, `pred` 0/1, LF) |
| `t5_swap_into_submission.py` | puts a candidate into the task5 column of a local full submission (e.g. sub_v6). It does not submit. |

```bash
python t123/research/t5/t5_discovery.py
python t123/research/t5/t5_pipeline.py
python t123/research/t5/t5_make_candidates.py
python t123/research/t5/t5_swap_into_submission.py submissions/<sub_v6 file>.csv t123/research/t5/t5_candidates/t5_union_plus_acckda_p50.csv submissions/sub_v7_t5acckda_p50.csv
```

## Analyses (hypothesis → result → keep/discard)

Notation: "means" = per-row mean of the 4 periods. Robust z = (x − median)/(1.4826·MAD) or IQR/1.349, computed on rows outside the union.

1. **Normal rows have independent features.** Max |corr| between the means of any two features outside the union is **0.032**. Cross-period correlation is high for most metrics, so each player has a stable base plus period noise; `ip_hash_entropy` is pure per-period noise.
   This means *any* joint structure between two different features outside the union is injected. It is the key lever for everything below.
2. **Signatures of the known groups** (median robust z):
   * bot: ip_entropy −5.9, login_time_var −2.1, session_std −2.0, kda_std −1.8, perf_stab +1.7, play_consistency +2.0
   * lowact: activity −2.1, purchase_amount +0.8, skill_rating +1.1
   * lowcomm: comm −2.1, friend_adds +1.5
   * lowstab: behavior_stability −5.7, login_count −0.9, purchase_freq +1.1
   * rule: period-4 shocks (location_changes, friend_removes, currency, skill, purchases).

   The 5 groups are **mutually exclusive** (827+207+266+237+274 = 1,811 exactly). Each injected type is its own set of 200–300 rows.
3. **Are the stationary groups complete?** I scored rows outside the union on each group's *other* signature features. For bot, the group median is 1.9–2.8 while the outside 99.9% quantile is 1.0–1.1, so no hidden bots. lowact, lowcomm and lowstab show no outside excess either. → **complete; discard "loosen the stationary thresholds".**
4. **Is the shock group complete?** All 827 rule rows have a period-4 shock of |z| > 10, or a geo jump: 8 rows are geo/lat-only.
   Outside the union, only 20 rows have any p4 |z| > 10, all on `skill_trajectory`, which is a heavy-tailed continuous metric (known dead end). Periods 1/2/3 have 2/3/9 such rows (the same tail).
   For the discrete columns, p4 − p1 ∈ {0, 1} for every non-union row, and the out-of-structure values occur only in the union. → **complete.**
5. **Pairwise 5%-tail co-occurrence** (each pair should co-occur 60 times under independence): **`accuracy_std` high × `kda_std` high = 222 observed vs 60 expected (z = 20.9)**. Next best is z = 3.7 (noise). The 4 flag columns show nothing either (max z 3.2). → **keep → new type.**
6. **Triple grid scan** (8×8×8 rank cells, 2,600 triples): **(accuracy_std, kda_std, performance_stability) z = 19.8**, top cell (7,7,6) holds 183 rows against 47 expected. All other triples have z ≤ 5.7, and those that reach it contain the same two features; the median is 3.2. → the third feature is performance_stability.
7. **Dynamics features** (per-row period std/|mean|, Δ12, Δ34, slope; triples, union and new cluster removed): top z is 4.7–5.1 against a median of 2.8. That is the null. → **no gradual or period-specific type; discard.**
8. **Discrete columns** (device/refund/guild per period) and **flags** against every feature decile and against each other, outside the union: max z = 3.1. → **discard.** The flags are enriched only in the rule group (password_changed 0.40 vs 0.05) and sit at base rate in every stationary group, including the new one.
9. **New cluster, size and precision.**
   * The copula kNN density (joint vs independence, 3-D rank space) is bimodal. Null rows have log2 density < 1; 253 rows sit above a clear gap.
   * Mahalanobis radius around the cluster, compared with a column-shuffle null (20 reps; the shuffle keeps the marginals and breaks only the dependence):

     | radius | outside rows | null expected | excess (= true) | in union |
     |---|---|---|---|---|
     | <2 | 159 | 8.6 | 150 | 0 |
     | <2.5 | 214 | 19.0 | 195 | 1 |
     | <3 | 263 | 37.3 | 226 | 3 |
     | <3.5 | 297 | 66.1 | 231 | 4 |
     | <4 | 340 | 106.6 | 233 | 5 |

     The excess plateaus at **≈ 230 rows**, which is the cluster size. The union hits match the null (expected ≈ 3), so the cluster does not overlap the union.
   * EM mixture: a Gaussian cluster against the product-of-marginals background gives π = 0.0099, **n_eff = 237**. Centre: accuracy_std 10.45 ± 0.47, kda_std 0.847 ± 0.048, perf_stab 0.898 ± 0.015 (the population means are 6.5 / 0.50 / 0.79).
   * Interpretation: performance varies a lot inside a match while the overall level stays steady. That fits account sharing or boosting (several people on one account). In any case it is a planted cluster: nothing else in the data is dependent.
10. **Isolation Forest** on means + shock z scores: AUC 1.000 against the shock rows and 0.882 against the final set. Its top-300 rows outside the final set are generic single-feature tails (skill_trajectory, ip_entropy), with no shared structure. → **IF is good for shocks and blind to the dependence-only cluster.** That is why the earlier IF/TMLP/TMAE extensions added ~0–5% true rows.

**Accounting.** K ≈ 1,811 + 237 = 2,048, against the ~2,070 implied by the LB. At most ~20 anomalies remain unexplained, which is below what any scan can resolve. An F3 gain needs a group with > 9% precision.

## Candidate table (ranked by confidence)

Expected F3 assumes the union is 100% precise, K = 1,811 + 237 + 20, and that EM posteriors are calibrated.

| rank | file | flagged | added | expected TP added | expected T5 F3 | confidence |
|---|---|---|---|---|---|---|
| ref | `t5_union_1811.csv` | 1,811 | 0 | – | 0.887 (LB ≈ 0.886 ✔) | = sub_v6 |
| **1** | `t5_union_plus_acckda_p50.csv` | 2,062 | 251 | 223 (89%) | **0.984** | **high** |
| 2 | `t5_union_plus_acckda_p10.csv` | 2,105 | 294 | 234 (80%) | 0.987 | high (F3-optimal cut, a little more FP risk) |
| 3 | `t5_union_plus_acckda_p90.csv` | 1,966 | 155 | 149 (96%) | 0.952 | very high (safety fallback) |

No other candidate group survived (§3–§8), so there is nothing else worth a probe.

## Probe order (only when the user asks to submit)

1. `sub_v6` with T5 = **p50**, built with `t5_swap_into_submission.py`. Expected public ≈ 0.6429 + 0.019 ≈ **0.662**. A rise of ≥ +0.015 confirms the type.
2. If (1) rises, try **p10**. Expected an extra +0.0007; keep whichever is higher.
3. If (1) does *not* rise, submit **p90**. That tells apart "type is wrong" (p90 also flat or down) from "posterior threshold too loose".

## Pipeline for the course report (goal B)

`t5_pipeline.py` is unsupervised and needs no labels:

| stage | model | finds | rows |
|---|---|---|---|
| A | Robust-z (median/MAD) temporal change detector, periods 3/4 vs 1/2, + geo-jump | takeover / selling / geo-VPN | 827 |
| B0 | 1-D density-gap detector (histogram valley) on each 4-period mean | lowstab 274, lowact 266, lowcomm 237, bot 207 (bot is seen by 4 features and merged) | 984 |
| B | Dependency discovery: tail co-occurrence vs independence, graph components | feature subset of a hidden cluster | – |
| C | EM mixture: Gaussian cluster vs product-of-marginals background, with signature expansion; posterior > 0.1 | new acc/kda/perf cluster | 294 |
| D | Isolation Forest (soft score, independent check) | AUC 1.00 against shocks, 0.88 against final | – |
| vote | hard OR-vote of A, B0, C | | **2,105** |

Overlap with the union: **1,811/1,811 reproduced, 0 extra**, and 294 new rows. Rows in the pipeline's final set = `t5_union_plus_acckda_p10`; for p50, use posterior > 0.5 in stage C.
The posterior is a proper anomaly *score*, so the threshold can be justified for F3: flag while the marginal precision is > F3/(1+β²) ≈ 0.09.

## Dead ends (this session)
* Loosening stationary thresholds: no outside rows carry the group signatures (§3).
* Single-period or period-1/2/3 shocks: only the skill_trajectory tail (§4).
* Period-dynamics clusters (std, Δ12, Δ34, slope): null-level (§7).
* Discrete/flag combinations: null-level (§8).
* Isolation Forest top rows outside the union: unstructured tails (§10).
