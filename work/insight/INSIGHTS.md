# Insight hunt on Tasks 1, 2, 3, 5 (16 Nov 2026, evening)

Goal: look for structure that the earlier research rounds had not used. Nothing was submitted. Every script is in `work/insight/`
and runs from the repo root. Summary: no new score source in T1, T2 or T3 (all three sit at the noise floor of their generators);
T5 gains one row and an exact anomaly count.

## Cross-task and structural checks (`a01_crosstask.py`, `a02_structure.py`)
- The four tabular tasks share the same player ids (P000000..P129999, about 80% overlap between any two files), but they are
  independent draws: T2 segment vs T3 segment of the same player agree at chance (28%), cheater rate is flat across T2/T3 segments,
  and same-named columns (account_age_days, friend_count, ...) have correlation 0.00 across tasks. No cross-task join helps.
- No id or row-order signal in T1, T2 (new data) or T3: Spearman |rho| < 0.005, lag-1 autocorrelation 0.00, block means at the
  binomial level. No train/test duplicate rows. No decimal-precision artefact. The 13k unlabeled T1 rows are missing at random.

## T1: the generator is "uniform boxes per sub-population + Gaussian skill features" (`b01`, `b02`, `b03`)
- Sub-populations (age observed):
  | group | age (days) | friends | reports | level | crosshair |
  |---|---|---|---|---|---|
  | legit, new | 1-110 | U{0..49} | U{10..29} | U(28,100) | U(.82,.96) |
  | legit, regular | 120-1000 | U{45..249} | U{5..19} | same | same |
  | legit, veteran | 200-1500 | U{100..349} | U{5..24} | same | same (game_sense +0.04) |
  | cheater, young | 15-180 | U{10..89} | 2/3 U{4..15} + 1/3 U{4..19} | U(25,96) | U(.80,.94) |
  | cheater, old | 200-1000 | U{60..269} | same | same | same |
- Skill features are near-Gaussian with a small mean shift. Cheaters are slightly WORSE than legit players on every skill feature
  (d = 0.08-0.24 sd: KD 4.46 vs 4.68, headshot 45.4 vs 48.1, game sense .801 vs .825), and legit "new" accounts are the best
  (KD 5.6, headshot 53: smurf-like). There is no "cheaters aim better" signal in this data.
- Inside an overlap cell (age 200-1000, friends 100-249, reports 5-19; 6,006 complete rows) LDA on the 9 skill features gives AUC 0.619,
  QDA 0.606, quadratic logistic 0.615: the class difference is a pure mean shift (no covariance or mixture structure). The final GBM
  reaches 0.699 in the same cell because the step-shaped pmfs of reports and friends still carry information there.
- Does the GBM waste the linear skill signal? Cross-fitted logistic "skill logits" (global and per age group; alone AUC 0.640 / 0.693)
  added to fe_v2, LightGBM screening setup, seeds 42 and 7: AUC +0.00035 / +0.00020, nested binary F2 -0.00006 / +0.00015,
  nested macro F2 -0.00016 / -0.00059. No gain: the GBM already uses the linear part. T1 stays at its ceiling.

## T2: nothing beyond the known 2x2 latent + 3% label noise (`d01`, `d02`)
- Class = (engagement low/high) x (spending low/high). `engagement_level` and `loyalty_tier` are noisy read-outs of the two bits.
- Rows that look inconsistent inside a class (class 0 with engagement "Veteran", or with loyalty "Diamond") have the feature profile
  of the other class (+0.4 to +1.2 sd): they are the 3% flipped labels, already modelled by the EM-NB.
- Pooled within-class correlations of the 16 main features are all below 0.013, so naive Bayes is the right model family.
  Region, platform, device, payment, language, account_status, player_type_tag and skill_tier are class-independent.

## T3: exact shape of the generator and where the error comes from (`c01`-`c05`)
- y = 0 with probability 1 - p(x); otherwise y = prev_month_spending x base(segment) x step factors x noise.
  - base ratio (no premium, vip 0, normal month): segment 0 1.10, 1 1.63, 2 1.82, 3 3.04.
  - step factors (multiplicative, same in every segment): premium x1.52; vip 1/2/3/4/5 about x1.12/x1.23/x1.6/x1.75/x1.9;
    each extra game played x1.07; month 0 and 11 x1.26, months 5 and 6 x1.16; days_since_last_purchase < 15 x1.16, > 45 x0.80;
    purchase_frequency in steps (about -20% below 5, +16% above 17).
  - noise on the ratio scale is a trapezoid (flat 0.86-1.10, zero outside 0.66-1.38): a product of two bounded uniform draws.
    Its log-sd is 0.15 in every segment; the learning curve of the residual is flat (0.158 at 5k rows -> 0.152 at 38k).
  - historical_spending = prev x U(2.49, 3.35) in every segment, so it adds nothing once prev is known.
  - whale "spike": 266 payers (1.7% of whales, 0.26% of rows) have y redrawn in about 90k-290k, independent of prev (cap 300,000).
    Not predictable (AUC 0.58, and that comes from the detection rule itself).
  - purchase_frequency and days_since_last_purchase are two read-outs of one latent purchase tier: frequency above 19 goes with
    recency 0-15 days, 10-14.5 with 0-45 days, 3.6-9.7 with 15 days or more, below 3.6 with 45 days or more (corr of logs -0.64).
    The recency / frequency steps of the multiplier and of p(pay) are this tier.
  - p(pay) is also a step function: segment (0.27 / 0.50 / 0.67 / 0.91 without premium), premium, vip >= 3,
    days_since_last_purchase <= 15, purchase_frequency > 15. Month has no effect on p(pay).
- New exact identity: `is_premium_member = (vip_status >= 3)` (0 exceptions in 97k rows; the T3 code does not use it). It fills
  premium for 5,715 train rows and bounds vip for 5,856. Proxy test (`c06_t3_premium_fill.py`, LightGBM, 2 x 5 folds): payer
  mean |log residual| on the premium-missing rows 0.1460 -> 0.1399 (-4%), all payers 0.14075 -> 0.14028; pay log-loss 0.55316 ->
  0.55293. But on the real recipe (`c08_t3_fill_ab.py`: LightGBM p + CatBoost depth-4 multi-quantile, one seed, same folds) the fill gives
  0.79499 vs 0.79506 without it (-0.00007, 2 of 5 folds up): CatBoost already reads premium off vip. Not adopted.
  The column scan (`c07_t3_identity_scan.py`) finds no other identity: month, primary_game, competitive_rank, platform and
  payment_methods_used cannot be predicted from the other columns; tournament_participation = 1 implies segment 1 and
  owns_limited_edition = 1 implies segment 2 or 3 (the segment imputer already has both columns).
- Noise-shape check on the real recipe (`c09_t3_pooled_noise.py`): replacing the 11 fitted quantiles by "fitted median x pooled
  residual quantiles" gives 0.79499 (pooled) / 0.79500 (per segment) vs 0.79506; geometric mean of both 0.79504-0.79506. The two
  are the same model within 0.0001, which confirms the noise is one fixed shape for every payer. Not adopted.
- Error budget of the best OOF (0.79529):
  | source | rows | share of total absolute error |
  |---|---|---|
  | payer, multiplicative noise | 37,955 | 49.0% |
  | y = 0 but prediction > 0 (coin flip of paying) | 13,454 | 32.9% |
  | whale spike (y redrawn) | 266 | 15.4% |
  | payer but prediction 0 | 15,730 | 2.7% |
  Whales (segment 3) hold 87% of total spending and 79% of the error. Score without the 266 spike rows would be 0.8135; with every
  zero known in advance 0.8527. All three sources are random draws of the generator, so 0.795 is the floor for this metric.

## T5: the old file gives the generator's exact layout; one row added (`e01`-`e07`)
- In the OLD test file (the one with the id leak) the anomalies sit in fixed id blocks: 620 shock rows (takeover 310 + selling 310),
  207 bot, 207 VPN, then 1,036 "stationary" rows with a random type among lowact / lowcomm / lowstab / acckda. Total 2,070 = 8%.
- The NEW file has the same layout: shock 827 (splits 310 / 310 / 207), bot 207, and four stationary boxes with 1,018 certain rows
  (lowact 265, lowcomm 263, acckda 244, lowstab 246 visible). So 18 anomalies are hidden behind missing values:
  - row 25766 (P005388): activity_level missing in all 4 periods (the only such row in the file), purchase_amount and skill_rating
    inside the lowact box in all observed periods. Posterior about 0.28 -> worth adding under F3 (threshold 0.0995).
  - 17 lowstab rows with missing behavior_stability among 128 candidates (the old prior was 14).
- A classifier "visible lowstab vs normal rows passing the same box" (CV AUC 0.89) only uses login_count and purchase_frequency, the
  same information as the existing posterior. The hidden rows cannot be pinned down further.
- Expected F3 under this accounting: current file (2,104 rows) 0.99681; + row 25766 0.99690; + prior 17 (2,115 rows) 0.99698;
  perfect would be 1.0 (+0.0006 LB). Files: `t5/preds/v2data/t5v2_union_box_lsnan_la.csv` (2,105 rows),
  `t5/preds/v2data/t5v2_union_box_lsnan17_la.csv` (2,115 rows), `submissions/FINAL/sub_v66_clean_Lv3_T5acct.csv`
  (= sub_v64a with the 2,115-row T5; 11 T5 rows changed; expected LB change +0.00003, below what the public LB can show).

## What this means for the score
T5 public F3 is about 0.997 and T1/T2/T3 are at the floors of their generators, so with weights 0.2 the public score of a file
implies its T4 macro-F1: T4 = 5 x LB - (0.839 + 0.758 + 0.795 + 0.997). v25a 0.900, v37n 0.936, v58f 0.957. T4 is the only task
with headroom, and at most about 0.009 LB of it is left.
