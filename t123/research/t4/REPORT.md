# T4 (game screenshots, 5 classes, macro-F1): prior-shift study, v4 training script, label files

Base model: `v3_5f` = ConvNeXt-Tiny DINOv3, 5 folds, OOF macro-F1 **0.98886** (`t123/preds_final/t4/preds/`).
Class order everywhere = datacard / alphabetical: 0 `7-11k`, 1 `fiveN`, 2 `freefried`, 3 `pube`, 4 `roblock`.
Nothing was submitted. No test labels exist and none were used. The only use of test data is the **unsupervised**
prior estimate in §A, computed from the model's **test predictions only**.

| file | what |
|---|---|
| `t4_prior_shift.py` | §A: calibration, test-prior estimates (EM / BBSE), honest OOF shift simulation, nested logit bias. `python t123/research/t4/t4_prior_shift.py --reps 300` (~2.5 min CPU) |
| `results_prior_shift.json`, `log_prior_shift.txt` | all numbers from that run |
| `t4_test_proba_{ts,bcts}_em.npy` | EM-corrected test probabilities (kept for reference; **not recommended**) |
| `task4_kaggle_v4.py` | §B: improved Kaggle GPU training script (not run here) |
| `t4_blend_runs.py` | §B: blends v3 + v4 runs on OOF (nested-CV weights) → test proba + csv |
| `t4_make_label_csv.py` | §C: builds the two string-label files below |
| `t4_test_str.csv` | §C: `id,pred`, labels exactly as in train.csv (`7-11k`) |
| `t4_test_str_711K.csv` | §C: same, `7-11K` (capital K, slide page 30) |

---

## A. Prior-shift correction: **not adopted** (simulated gain ≈ 0)

### A1. Facts
| | 7-11k | fiveN | freefried | pube | roblock |
|---|---|---|---|---|---|
| train prior | .394 | .140 | .177 | .188 | .101 |
| test argmax share (v3_5f) | .152 | .260 | .152 | .259 | .178 |
| test mean probability | .133 | .256 | .157 | .255 | .199 |

The test is also **less confident** than the OOF (max-prob quantiles q01/q05/q10/q50: OOF .71/.80/.81/.91, test .40/.52/.63/.91).
So besides label shift, there is some covariate shift (harder or different-looking screenshots).

### A2. Calibration (needed for EM)
The model was trained with label smoothing 0.1, an EMA and a fold average, so its raw softmax is **under-confident**.
Cross-fitted (5-fold) OOF NLL: raw 0.1527 → temperature scaling (TS, T≈0.50) 0.0500 → bias-corrected TS (BCTS) 0.0435.
- TS does not change the argmax.
- BCTS adds per-class biases, and on its own it **lowers** macro-F1 (nested OOF: raw 0.98886 → BCTS argmax 0.98868; simulation below: −0.0011).

### A3. Test-prior estimates (unsupervised, test predictions only)
| estimator | 7-11k | fiveN | freefried | pube | roblock |
|---|---|---|---|---|---|
| EM (Saerens), TS-calibrated | .135 | .268 | .145 | .263 | .189 |
| EM, BCTS-calibrated | .155 | .256 | .146 | .273 | .171 |
| BBSE hard (confusion matrix) | .149 | .260 | .152 | .260 | .179 |
| BBSE soft, BCTS | .165 | .246 | .153 | .269 | .167 |
| EM on **uncalibrated** probs | .027 | .311 | .133 | .275 | .254 |

All calibrated estimators agree within about ±0.015, so the shift is real (7-11k is 39% in train and about 15% in test).
EM on the raw, under-confident softmax gives an absurd answer (7-11k 2.7%), which shows calibration is required.

### A4. Honest simulation on OOF (300 resamples per scenario)
Design:
- Split the OOF rows into stratified halves A and B. Fit calibration (and the logit bias) on **A only**.
- From **B**, draw 25,889 rows by class with the target prior = EM-BCTS estimate above.
- Run the whole procedure blind on the sample (EM on the sample, re-weight `p·q/π_train`, argmax). Score macro-F1 against the sample's true labels.
- Alternate A and B between reps.
- Scenario `+conf` also re-weights rows inside each class so the max-prob distribution matches the test. This is a stress test for "test is harder".
- `oracle` = re-weighting with the sample's *true* prior, which is an upper bound for any prior estimator.

Δ = macro-F1 minus the **raw argmax** on the same sample (mean ± sd over 300 reps; P = share of reps with Δ > 0):

| calibration / scenario | raw macro-F1 | EM | BBSE hard | BBSE soft | oracle prior | logit bias | EM + bias |
|---|---|---|---|---|---|---|---|
| TS · label shift | .98862 ± .0019 | +.00008 ± .00026 (P .58) | +.00010 (P .60) | +.00010 | +.00010 | −.00025 (P .15) | −.00014 |
| TS · label shift + low-conf | .96424 ± .0058 | −.00014 ± .00070 (P .44) | −.00013 | −.00013 | −.00015 | −.00112 (P .09) | −.00127 |
| TS · no shift | .98889 | −.00005 | −.00001 | −.00001 | .00000 | +.00014 (P .69) | +.00021 |
| BCTS · label shift | .98853 | −.00002 (P .47) | −.00002 | −.00002 | −.00002 | −.00132 | −.00025 |
| BCTS · label shift + low-conf | .96438 | −.00055 (P .12) | −.00056 | −.00061 | −.00042 | −.00466 | −.00145 |
| BCTS · no shift | .98888 | −.00013 | −.00013 | −.00013 | −.00013 | −.00024 | −.00023 |

Reading:
- The prior estimate is accurate (L1 error 0.005–0.02). Even so, **re-weighting with the exact true prior gains at most +0.0001**, which is inside the noise.
  The classifier is so confident that almost no decision sits close enough to a boundary for a ×0.4 / ×1.9 prior ratio to flip it.
  When it does flip one, it is about as often wrong as right.
- In the "harder test" stress test the correction becomes slightly **negative**.
- **Caution, an earlier version of this analysis looked positive (+0.0011 / +0.0033).** That was an artifact: Δ was measured against the BCTS-calibrated argmax, which is itself 0.001–0.004 worse than the raw argmax. EM only won back what BCTS had lost. Measured against the raw argmax there is no gain.
- On the real test, TS+EM would change **464 labels (1.8%)**, mostly 7-11k → other classes (the change matrix is in the JSON).
  Those are exactly the low-confidence test images, so the simulation suggests the net effect is ~0 ± 0.001. Since the public LB does not score T4 (CLAUDE.md §4), there is no way to check it.

### A5. Nested per-class logit bias for macro-F1
Coordinate search on log-probabilities, fitted on 4/5 of the OOF and scored on 1/5:
raw argmax **0.98886** · TS argmax 0.98886 · TS + bias **0.98877** · BCTS argmax 0.98868 · BCTS + bias 0.98856.
In the simulation the bias costs −0.0002 to −0.0011 under the shifted scenarios.
The fitted biases are unstable between halves (fiveN −1.8 in both, others flip sign), which is the classic sign of over-fitting ~350 errors.

**Decision A:** keep the plain argmax of the v3_5f fold-average (no calibration, no prior correction, no bias).
The criterion "keep only if the simulated gain is clearly positive" is not met.

---

## B. `task4_kaggle_v4.py`: improved Kaggle GPU script (not run; no GPU or images here)

**Not executed:** this sandbox has no GPU, no images and no torch, so the script was only syntax-checked (`py_compile`); the augmentation and flip maths were checked by hand. Expect small fixes on the first Kaggle run (for example, batch size if a T4 runs out of memory).

Run it once per backbone (`T4_BACKBONE=cnxt_s_d3 | eva02_s | cnxt_t_d3`). It uses the same 5 folds as v3 (`StratifiedKFold(5, shuffle, 42)`), so OOFs from all runs can be blended. Metadata needed:
`enable_gpu: true`, `enable_internet: true`, data source = the task4 image dataset (+ the competition, for `sample_submission.csv`), Kaggle secret `HF_TOKEN` (DINOv3 is gated).
The token is read only through `kaggle_secrets.UserSecretsClient`; nothing is hardcoded. EVA02 does not need the token.

| change | v3_5f | v4 | why |
|---|---|---|---|
| input | 192×352, rotation ±30° with black corners, scale 0.85–1.2 | 256×480 (T) / 224×416 (S) / 224×420 (EVA), whole frame, bicubic; rotation ±8°, scale 0.92–1.08, shift 4%, reflection padding | HUD / logo details sit at the borders; heavy rotation-and-crop removed them |
| flips | none | hflip p=0.5 in training, plus an hflip TTA view | cheap TTA; each view is saved separately, so you can check on OOF that hflip does not hurt (set `HFLIP_P=0` if it does) |
| robustness | colour jitter, channel permute, invert, noise | colour jitter, grey, **down/up-sample degradation**, noise, random erasing | the test is less confident than the OOF (§A1), which suggests lower-quality screenshots |
| imbalance | class-weighted CE | `sqrt` inverse-frequency **sampler** (train labels only), plain CE, label smoothing 0.05 | balanced batches help macro-F1. A `prior` mode exists but is OFF: the test mix is only known from test predictions, and the rules forbid using test data for training |
| schedule | 14 epochs, **best epoch chosen on the validation fold** | 18–20 epochs, EMA 0.9995, cosine to 1%, **final EMA weights** | v3's OOF was slightly optimistic because the epoch was picked on the same fold it reports. v4's OOF is honest |
| backbones | ConvNeXt-T DINOv3 | + ConvNeXt-S DINOv3, + EVA02-S (MIM in22k → in1k, ViT) | two different architecture families give the most diverse ensemble |
| TTA | none | views `plain`, `hflip`, `up` (×1.125); default average plain+hflip | view choice is made on OOF |
| Kaggle limits | — | time guard 11 h; `T4_FOLDS=0,1` lets you split the folds over sessions | ConvNeXt-S / EVA02 at this resolution may not fit 5 folds in one 12 h T4 session |

Outputs (same format as v3, plus per-view files): `preds/v4_<bb>_oof.npy`, `_oof_<view>.npy`, `_test_fold{k}.npy`, `_test_fold{k}_<view>.npy`, `task4_v4_<bb>_test_proba.npy`, `task4_v4_<bb>.csv`, `v4_<bb>_log.json`.

Blending (local, CPU):
`python t123/research/t4/t4_blend_runs.py --run v3_5f=t123/preds_final/t4/preds --run v4_cnxt_s_d3=<dir> --run v4_eva02_s=<dir> [--views plain,hflip]`
It searches log-prob blend weights on a simplex grid and reports a **nested 5-fold** OOF score next to the equal-weight blend, then writes `blend/t4_blend.csv` and `blend/t4_blend_test_str.csv`.
I tested it with v3 alone (it reproduces v3 exactly) and with a synthetic noisy second run (it correctly gives that run weight 0).
v3's OOF is slightly optimistic (best-epoch selection), so its blend weight will be a bit inflated. Prefer the equal-weight blend unless the nested score is clearly higher.

Suggested order (about 30 GPU-h per week): 1) `eva02_s` (most diverse), 2) `cnxt_s_d3`. Then blend with v3_5f.
Expected gain: v3 makes ~350 OOF errors out of 31.5k, so realistically +0.001–0.003 OOF. The test is harder than the OOF, so a robust ensemble matters more there.

---

## C. Label format files
- `t4_test_str.csv`: `id,pred`, 25,889 rows, `id = ANS00001..` in test.csv order, LF. Labels are exactly the train.csv strings (`7-11k, fiveN, freefried, pube, roblock`).
  Predictions = v3_5f fold-average argmax, identical to `pred_str` in `task4_v3_5f.csv`.
  Counts: fiveN 6,723 · pube 6,708 · roblock 4,607 · freefried 3,927 · 7-11k 3,924.
- `t4_test_str_711K.csv`: the same, with `7-11k` written `7-11K` (3,924 rows), as spelled on slide page 30.
  Slide page 36 (the sample submission) shows string labels (`freefried`, `fiveN`) in the task4 column, so the private LB may expect strings. Which spelling of 7-11k it expects is unknown.
  The public LB ignores T4, so this cannot be probed. Keep both files for the final pick.

## Recommendation
1. T4 final = **v3_5f plain argmax** (`t4_test_str.csv` strings, or v6's numeric column), with **no prior-shift correction and no logit bias**.
2. If GPU time allows, run `task4_kaggle_v4.py` for `eva02_s` and `cnxt_s_d3`. Blend with `t4_blend_runs.py` and switch only if the nested/equal-weight OOF beats 0.9889 and the per-view numbers look sane.
3. For the string-vs-number question: the slide's sample submission uses strings in the task4 column.
   If the team decides the private LB expects strings, put `t4_test_str.csv` (or the 711K variant) into the task4 column of the final file. That changes only T4, which the public LB does not score.
