# Task 5: Account Anomaly Detection (Methodology)

## 1. Problem and metric
- **Data:** test only (25,889 accounts), with no labels and no training set. Each account has ~29 behavioural metrics measured in 4 consecutive periods (`_1` … `_4`), plus static fields (`account_age_days`, `behavior_stability`, security flags).
- **Metric:** F3 of the anomaly class. Recall weighs 9× precision, so adding a row pays off once its probability of being anomalous exceeds about F/(1+β²) ≈ 8%.

## 2. Key EDA finding
Anomalies are not a smooth tail. They form **compact groups that are clearly separated from normal accounts**:
- sudden period-3/4 shocks
- one-dimensional clusters separated by empty histogram gaps
- one joint 3-D box

We therefore detect each group with an explicit, interpretable rule instead of ranking a generic anomaly score.

**Approach: unsupervised, rule-based detection guided by EDA.** No model is trained to predict anomalies (there are no labels). Each detector is a statistical test or threshold:

| Detector | Technique | Model? |
|---|---|---|
| Temporal shock | robust z-score (median/MAD) of the period-3/4 change vs the period-1–2 baseline; geo distance | no (statistic + threshold) |
| Gap clusters | histogram gap scan, cut at the empty bins | no (threshold) |
| Joint box | pairwise co-occurrence z-test, expanded to a 3-D box; confirmed by a Gaussian mixture (EM) | GMM used only as a check; the box is the rule |
| Hidden lowstab | naive-Bayes posterior (histogram likelihood ratio of 2 features + prior) | small probabilistic model fitted on the already-detected lowstab members, no labels |

Why rules instead of an anomaly-detection model:
- **No labels:** supervised learning is impossible, and an unsupervised score still needs a threshold chosen without feedback.
- **The models we tried were worse** (baselines in §6):
  - Isolation Forest catches the shock sub-types and bots but misses most of the stationary clusters and the acc/kda box.
  - A temporal masked autoencoder misses selling, bots, every stationary cluster and the box.
  - Both mix many normal accounts into their top ranks.
- **The groups have hard edges** (for example, no account has behavior_stability between 0.40 and 0.70), so a rule captures them exactly. Leaderboard probes confirmed that every row added outside the rule-based set lowered the score.

## 3. How each group was discovered
The guiding observation: **in normal accounts every pair of features is independent** (median |r| = 0.005; after removing the detected accounts the largest |r| is 0.026). Injected anomaly types break this in three visible ways: a sudden change over time, an isolated cluster on one feature, or two features that are extreme together. We searched for each pattern in turn, removed what was found, and searched the remainder again.

| Step | Search | What it revealed |
|---|---|---|
| 1. Time | For every metric: change = max over periods 3/4 of \|value − mean of periods 1–2\|, scaled to a robust z (median/MAD). Plus login distance from the period-1–2 "home" location. | Normal accounts end at about z ≈ 5, and a separate population has z > 6 on two or more metrics at once → **shock group**. Requiring ≥ 2 metrics removes single-metric noise. |
| 2. Shock sub-types | For shock rows, list which metrics move together and in which direction. | Three signatures: takeover, selling, VPN (table in §4.1). Check: only shock rows show ≥ 2 signature metrics beyond the members' 5% level (0 outside). Security flags, never used by the rules, confirm the split: password_changed is set in 99% and mass_item_sale in 69% of takeovers, and vpn_usage is set in 100% of the VPN group (base rates 5%, 2% and 15%). |
| 3. One feature | For each feature (4-period mean, or static field), take an 80-bin histogram of the non-shock rows. Look for a run of ≥ 3 near-empty bins (≤ 2% of a typical bin) that cuts off a small side (50–2,000 rows). | Four isolated clusters: low activity, low communication, low stability, and one cluster that appears as the same 207 rows in 4 features (ip_hash_entropy, kda_std, login_time_variance, session_duration_std) → merged into **bot**. |
| 4. Two features | On rows not yet flagged, mark each feature's lower and upper 5% tail. Count how often two tails co-occur vs the independence expectation and turn that into a z-score. Pairs with z > 6 form a graph, and each connected component is a candidate type. | One strong edge: high accuracy_std × high kda_std (z = 20.9). Fitting a 2-component EM (Gaussian cluster vs independent background) and adding every feature whose cluster mean shifts > 0.6 sd adds performance_stability (and its lower period-to-period variability). The per-period min/max of confident members (posterior > 0.9) defines the **acc/kda box**. |
| 5. Missing values | The low-stability rule reads one static column, which is NaN for ~5% of accounts, so ~13 members must be hidden. | Among NaN accounts inside the members' purchase_frequency and age ranges we observe 128 accounts vs 109 expected by chance → the **hidden lowstab** posterior (§4.4). |
| 6. Completeness | Repeat the pairwise search over value bins of all features after removing every group. | Largest remaining z = 5, vs 20.9 for the acc/kda edge and > 100 for the bot cluster → no further group. |

## 4. Detectors (final set = union, 2,104 accounts)

| # | Detector | Rows |
|---|---|---|
| 1 | Temporal shock | 827 |
| 2 | Stationary gap clusters | 981 |
| 3 | Joint acc/kda box | 244 |
| 4 | Hidden lowstab (NaN-aware posterior) | 52 |

### 4.1 Temporal shock (827)
- **Baseline:** each account's mean of periods 1–2.
- **Flag** an account when any of these holds:
  - ≥ 2 metrics change in period 3/4 with robust z > 6 (median/MAD over all accounts);
  - the login location moves > 5° from the baseline;
  - |lat| > 90 (invalid coordinates).
- **Sub-types** found by their signatures:

| Sub-type | Rows | Signature |
|---|---|---|
| Takeover | 310 | currency_transfers, location_changes, device_count ↑; play_pattern_consistency ↓ |
| Selling | 311 | skill_rating, kda, accuracy ↑; reaction_time ↓; friend_removes ↑ |
| VPN | 198 | purchase amount/frequency ↑; location changes |
| Geo only | 8 | location jump without metric shocks |

### 4.2 Stationary gap clusters (981)
- **Method:** histogram the 4-period mean of every metric (and each static field), then scan for empty bins that split off a small cluster.
- **Clusters found:**

| Cluster | Rows | Rule |
|---|---|---|
| Bot | 207 | login_time_variance, ip_hash_entropy, kda_std and session_duration_std all low together |
| Low activity | 265 | mean activity_level < 0.345 |
| Low communication | 263 | mean communication_frequency < 0.169 |
| Low stability | 246 | behavior_stability < 0.40 (no account between 0.40 and 0.70) |

- **Coverage:** a full gap scan of every mean, std and static feature finds no further isolated cluster.

### 4.3 Joint acc/kda box (244)
- **Why a joint search:** this group cannot be seen in any single metric. It was found by the pairwise 5%-tail co-occurrence test (§3, step 4).
- **Box:** EM plus signature expansion added performance_stability. The box is the per-period min/max of the confident members:
  - accuracy_std ∈ [8.96, 12.03]
  - kda_std ∈ [0.70, 1.00]
  - performance_stability ∈ [0.85, 0.95]
- **Check:** a Gaussian EM mixture fitted to the same metrics confirms a component of ~0.94% of accounts, centred inside the box.

### 4.4 Hidden low-stability members (+52)
- **Problem:** ~5% of accounts have NaN `behavior_stability`, so lowstab members among them escape rule 4.2. The expected number hidden is ~13 (cluster size × NaN rate).
- **Candidates:** NaN accounts outside the union whose purchase_frequency ∈ [3, 6] and account_age ∈ [800, 1500], the ranges that contain all lowstab members.
- **Score:** a naive-Bayes posterior using two traits of the cluster, low mean login_count and high period-to-period variation of purchase_frequency, with prior = expected hidden members / number of candidates.
- **Selection:** posterior > 0.1, matching the F3 break-even of about 8%.

## 5. Validation without labels
- **Weight of T5:** leaderboard probes on the original release showed that setting T5 to all-zero dropped the public score by 0.177, so T5 carries real weight.
- **Precision outside the union:** every attempt to add accounts outside the union lowered the score. Looser thresholds (3,602 rows) gave −0.012, and shock z 4–6 with geo 2–5° (+356 rows) gave −0.0025. So the outside rows hold almost no anomalies and the union is near-complete.
- **Posterior cut:** the lsnan threshold was checked on the leaderboard. A cut of 0.15 lost 0.00012 and 0.05 was flat, so 0.1 was kept.
- **Re-release:** on the re-released data (12 Nov) the same pipeline recovered groups of almost identical size, and the joint-box search found no new group (largest remaining z = 5 vs > 100 for real groups).

## 6. Baselines and methods tried and discarded

| Method | Result |
|---|---|
| Isolation Forest (top 2,104) | catches takeover, selling, VPN and bots; misses most low activity (13% caught), low communication (18%), low stability (32%), the acc/kda box (10%) and all hidden lowstab members |
| Temporal masked autoencoder (top 2,104) | catches about half of the takeovers and a third of the VPN group; misses selling, bots, every stationary cluster and the box (≤ 15% each) |
| Gaussian-EM posterior instead of box | box has sharper edges; posterior variants superseded |
| Looser shock/geo thresholds | public LB −0.012 (3,602 rows) |

## 7. Output
- Binary prediction: 1 = anomaly for the 2,104 accounts in the union of §4.1–4.4, otherwise 0.
- **Code:**
  - `t5/research/t5_pipeline.py` (shock, gap scan, box)
  - `t5/research/t5_box_analysis.py`: checks that box rules reproduce each group, the acc/kda box bounds and chance rate, the completeness pair search (§3 step 6), the shock sub-type signatures (§3 step 2), and the hidden lowstab posterior (§4.4) that produces the lsnan rows
- **File:** `t5/preds/v2data/t5v2_union_box_lsnan.csv`
