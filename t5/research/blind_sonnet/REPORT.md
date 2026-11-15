# Blind T5 re-derivation (25,889 rows, no labels)

## Findings
Anomalies form 7 disjoint, sharply bounded groups, total 1,811 rows, zero overlap. All single-feature stationary cuts sit in empty gaps of 4,000-42,000x the median spacing. p-values are never used; the evidence is gap emptiness, multi-metric co-occurrence and flag purity.
base123 = nanmean of periods 1-3. lr(m) = log((x4+s)/(base123+s)), s = 0.05*std(x4). Geo distance = great-circle degrees from home = mean of period 1-2 (lat, lon).

| group | rule | rows | block-shuffle null |
|---|---|---|---|
| takeover_A | geo jump at p3 (>5 deg or |lat|>90) OR (lr purchase_frequency>0.9 AND lr purchase_amount_mean>0.55) OR (geo jump p4 AND (pf or pam up) AND no play drop); not perf_spike | 207 | strict co-occurrence g3&(pf\|pam)&vpn: real 171 vs null 0.2 (max 1) |
| takeover_B | (lr play_pattern_consistency<-0.3 OR geo jump p4>5) not in A or C | 310 | play&g4&pw: real 268 vs null 0.2 (max 1) |
| perf_spike_C | lr skill_rating>0.26 OR (skill_trajectory_4 - base123)>25 | 310 | sk&st: real 281 vs null 3.0 (max 5) |
| bot_lowvariance | mean123 of ip_hash_entropy<0.25 / login_time_variance<1.0 / session_duration_std<6.5 / kda_std<0.146 (all four coincide on 206-207 rows) | 207 | marginal rule, null = same count (no information); gap ratio 4k-17k |
| low_comm | mean123 communication_frequency<0.17 | 237 | gap ratio 6.4k |
| low_activity | mean123 activity_level<0.34 | 266 | gap ratio 4.2k |
| low_behavior_stability | behavior_stability<0.55 | 274 | gap ratio 42k |

Flag purity (independent confirmation, not used in the rules): takeover_A has vpn_usage=1 on 190/207 rows (17 NaN, 0 zeros); takeover_B has password_changed=1 on 301/310 (9 NaN, 0 zeros), mass_item_sale=1 on 196/310. Normal rows: vpn 15%, password 5%. Perf_spike_C and the stationary groups show no flag enrichment. 14-ish low_behavior_stability rows are probably missed because that column is NaN in 4.9% of rows. Leftover rows (outside the union) with >=2 big period-4 shifts (robust z>6): 0. Rows outside with >=3 of 7 C-votes: 0.
Half-subsample (x2) rule counts match the full counts within about +/-10% (A 202-220, B 290-322, C 280-330, bot 188-222, comm 232-246, act 250-286, beh 268-294).

Column-shuffle null (as specified) breaks the period-1..3 vs period-4 pairing, so the delta rules fire on thousands of rows (A 23,833; B 1,386; C 4,956). It is uninformative for the temporal rules. Counts for the static rules under it: bot 30, comm 2, act 5, beh 274 (static column unchanged). Counts are in out/rule_null.json.

## Detectors (features: base mean/std, period-3 and 4 deltas, geo distance; robust-scaled, clipped; scores in out/*_score.npy)
AUC / precision at top 1,811 are measured against the rule union (consistency, not truth).
| detector | AUC vs union | prec@1811 | real rows above null max (col / block) | stability |
|---|---|---|---|---|
| iforest | 0.865 | 0.551 | 406 / 82 | seed Jaccard 0.65, half-sample rho 0.88 |
| ocsvm (Nystroem 400 + SGD) | 0.858 | 0.533 | 0 / 831 | seed Jaccard 0.99, rho 0.999 |
| lof | 0.518 | 0.199 | 36 / 0 | rho 0.94 |
| pca_recon (15 PCs) | 0.800 | 0.393 | 2 / 1 | rho 0.997 |
| mahalanobis | 0.871 | 0.553 | 4 / 1 | rho 0.999 |
| gmm (8 diag on PCA20) | 0.803 | 0.454 | 1 / 0 | Jaccard 0.91, rho 0.97 |
| hdbscan (min_cluster_size 80, PCA15) | - | - | - | recovers takeover_B (268/310) and takeover_A (132/207) as clusters; others merged with normal |

Per-group AUC (iforest / ocsvm / mahalanobis): A and B and C about 0.98-1.0. bot 0.8-0.88. low_comm and low_activity 0.58-0.67 (they lie inside the marginal range, so density detectors do not see them). low_behavior_stability 0.77-0.87 for iforest/ocsvm/mahalanobis/pca. LOF fails (0.26-0.47 on A/B/C). The null-exceedance numbers are ambiguous: the col null inflates scores, the block null leaves marginal tails. The generic detectors find A, B and C but not the stationary groups, and give no decisive null separation. The rule groups were obtained by gap analysis and flag cross-checks, not by these detectors.

## Files
blind_groups.csv (row_index 0-based in test.csv order, group_name), *_score.npy (iforest, ocsvm, lof, pca_recon, mahalanobis, gmm, hdbscan, rules_score), det_eval.json, rule_null.json, cooc.json. Code is in the parent folder (final.py, fe.py, det.py, evaldet.py).
