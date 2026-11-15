# T4 (game screenshots, 5 classes, macro-F1): prior-shift study, v4 training script, label files

Base model: `v3_5f` = ConvNeXt-Tiny DINOv3, 5 folds, OOF macro-F1 **0.98886** (`t4/preds/v3_5f/preds/`).
Class order everywhere = datacard / alphabetical: 0 `7-11k`, 1 `fiveN`, 2 `freefried`, 3 `pube`, 4 `roblock`.
Nothing was submitted. No test labels exist and none were used. The only use of test data is the **unsupervised**
prior estimate in §A, computed from the model's **test predictions only**.

| file | what |
|---|---|
| `t4_prior_shift.py` | §A: calibration, test-prior estimates (EM / BBSE), honest OOF shift simulation, nested logit bias. `python t4/research/t4_prior_shift.py --reps 300` (~2.5 min CPU) |
| `results_prior_shift.json`, `log_prior_shift.txt` | all numbers from that run |
| `t4_test_proba_{ts,bcts}_em.npy` | EM-corrected test probabilities (kept for reference; **not recommended**) |
| `task4_kaggle_v4.py` | §B: improved Kaggle GPU training script (not run here) |
| `t4_robust_blend.py` | §E: robust ensemble-weight blending pipeline (per-member calibration, 6 blenders, nested shift-aware evaluation, conservative selection) |
| `t4_test_diagnostics.py`, `results_diagnostics.json`, `log_diagnostics.txt` | §D: label-free test-difficulty estimates, fold disagreement, uncertainty pairs, expected-F1 decoding |
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

Blending: use `t4_robust_blend.py` (§E).
v3's OOF is slightly optimistic (best-epoch selection), so v3 will look a little better than it is in any OOF-based weighting.

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

## D. Follow-up research ("Research More", `t4_test_diagnostics.py`, `results_diagnostics.json`)
Run: `python t4/research/t4_test_diagnostics.py --reps 100` (~6 min CPU). As before, it uses test **predictions only**.

### D1. The test is much harder than the OOF: the main T4 finding
| label-free signal | in-distribution (OOF) | test |
|---|---|---|
| all 5 fold models give the same label | ≈ 0.98 expected (OOF error 1.1%) | **0.834** |
| mean pairwise agreement of fold models | ≥ 0.978 expected | **0.913** |
| rows with max-prob < 0.6 | 0.5% | **8.6%** |

- **Lower bound (no assumptions):** for two models, error_i + error_j ≥ disagreement_ij. With a mean pairwise disagreement of 8.66%, the average **single-fold model is wrong on ≥ 4.3% of test rows**, versus 1.1% on OOF.
- **Generalization-Disagreement Equality** (Jiang et al., ICLR 2022): the disagreement between independently trained models ≈ test error. That gives ≈ 8.7% single-model test error.
- **Label-free accuracy estimators**, first validated on OOF resamples with known truth:

| estimator | bias, label-shift sim | bias, label-shift + low-conf sim | **estimate on test** |
|---|---|---|---|
| ATC (Garg et al., ICLR 2022) | +0.001 | **−0.005** (best) | **0.867** |
| calibrated confidence (TS) | +0.005 | +0.020 | 0.957 |
| posterior sampling (TS + EM), macro-F1 | −0.003 | +0.011 | 0.944 |
| DoC | +0.032 | +0.039 | 0.958 (biased, ignore) |

  Confidence-based estimators are over-optimistic as soon as the test is harder. ATC was the only one accurate under the stress test.
- **Best guess for the v3_5f ensemble on test:** accuracy / macro-F1 ≈ **0.87–0.94**, not 0.989.
  The ensemble is better than a single fold, so it lands between the ATC number and the ~0.91–0.95 implied by GDE.
  T4 therefore has real headroom on the **private** LB (the public LB ignores T4), worth about +0.05 T4-F1 × T4 weight.
- **Image-size check:** I checked one train and one test image from the team dataset (`kiatisakkk/task4-image-classification`; only the size was read, then the files were deleted). Both are **230×120**.
  The "200×100" on the Kaggle data page is stale, so a resolution mismatch is ruled out as the cause. The shift is in the image content: other maps, modes or sources.

### D2. Where the test uncertainty is (what v4 should fix)
| | top pairs |
|---|---|
| OOF errors (296) | 7-11k/roblock .15 · 7-11k/fiveN .14 · freefried/pube .14 · pube/roblock .11 |
| uncertain test rows (max-prob < 0.6) | **freefried/pube .17 · fiveN/freefried .15 · fiveN/pube .14 · fiveN/roblock .13** · 7-11k/* ≤ .09 |
| fold disagreement by predicted class | 7-11k .09 · pube .14 · roblock .18 · fiveN .20 · **freefried .23** |

7-11k (39% of train) is stable on test. The shifted classes are **freefried, fiveN and pube**, and their test images often fall *between* those three classes.
This pattern is not visible in the OOF errors. It points to test screenshots of those games that look unlike the training ones.
Consequences for v4: robustness and diversity matter more than OOF decimals. That means the milder geometry (keep the HUD), degradation augmentation, a second architecture family (EVA02) and the TTA views.
The script now prints test pairwise fold disagreement (v3_5f 0.0866) as a **diagnostic** to compare runs.
Do not tune on it: it is computed on test images, and the rules say "no test usage for training purposes". Ask the TA before using it for model choice.

### D3. Fold ensemble: arithmetic vs geometric mean
Single fold models agree with the ensemble on 94.4–95.1% of test rows. Geometric instead of arithmetic averaging changes 142 test labels.
This cannot be validated on OOF (each OOF row comes from one model), so keep the arithmetic mean.

### D4. Expected-macro-F1-optimal decoding: not adopted
I chose per-class log-weights to maximise the posterior-*expected* macro-F1 on the unlabelled set, with TS posteriors or TS + EM posteriors, and used the same honest OOF simulation (100 reps).
- Δ vs raw argmax, label shift: TS +0.00003 ± 0.0001 (P>0 .51); EM +0.00008 ± 0.0003 (P .64).
- Δ vs raw argmax, label shift + low-conf: TS −0.00015 (P .00); EM −0.0001 ± 0.0007.
- On the real test it changes 0 labels (TS) or the same 464 as EM (EM).

No gain, the same conclusion as §A. With this model, the decision rule is not where the error comes from.

### D5. Competition pages (read through the Kaggle connector, read-only)
- The Overview defines T4 as 0 PUBE, 1 Free Fried, 2 FiveN, 3 Roblock, 4 7-Eleven Knight. That matches v6's numeric column.
- The local `sample_submission.csv` has numeric example values in `task4` (0.0/1.0/2.0) in only 4 rows; the other rows are blank. Slide 36 shows strings. So the format is still ambiguous, and keeping v6 numeric plus the string files (§C) as alternatives remains right.
- The data page also says "pipe-separated" (`id|task1|…`), which contradicts the comma format that is known to work. Ignore it.
- Rules §5 say "**No test set probing**: do not attempt to reverse-engineer or obtain test set labels". The team should keep this in mind for the leave-one-task-out LB probes planned in CLAUDE.md §5.
- There are no discussion topics in the forum.

### D6. Literature behind the choices
- **Saerens et al. 2002 (EM); Lipton et al. 2018 (BBSE); Alexandari, Kundaje & Shrikumar, ICML 2020:** EM with bias-corrected calibration is hard to beat for label shift. So our "no gain" is a strong negative result, not a weak method. It also explains why EM on uncalibrated probabilities failed (§A3).
- **Garg et al., ICLR 2022 (ATC):** used in D1.
- **Jiang, Nagarajan, Baek & Kolter, ICLR 2022 (Generalization-Disagreement Equality):** used in D1. Kirsch & Gal (TMLR 2022) note that it relies on ensemble calibration, which degrades as disagreement grows. Hence it is shown as an estimate, next to the assumption-free lower bound.

---

## E. Robust ensemble-weight blending pipeline (`t4_robust_blend.py`)
It replaces the earlier `t4_blend_runs.py`, which used one F1 grid with no calibration and no shift check.
```
python t4/research/t4_robust_blend.py --run v3_5f=t4/preds/v3_5f/preds \
       --run v4_eva02_s=<dir> --run v4_cnxt_s_d3=<dir> [--views plain,hflip] [--per-view] [--out t4/research/blend]
python t4/research/t4_robust_blend.py --selftest      # pipeline check with synthetic members (~1 min)
```
Steps:
1. **Load members.** A member is a run, or (with `--per-view`) a TTA view of a run. Folds from several Kaggle sessions are merged by giving the same TAG twice. The script checks that every OOF row and all 5 test folds are present.
2. **Per-member temperature scaling** on OOF. This matters because v3 (label smoothing 0.1) and v4 (0.05) differ in sharpness. In a plain probability average, the sharper model would dominate regardless of quality. Self-test temperatures: 0.50 / 1.18 / 2.61.
3. **Candidate blenders**, all log-linear pooling with simplex weights:
   - `equal`;
   - `nll`: log-loss-optimal weights;
   - `nll_bag`: the same, bagged over 20 bootstrap resamples;
   - `caruana`: greedy forward selection with replacement, bagged ×10 (Caruana et al. 2004);
   - `shrink`: ½ bagged-nll + ½ equal;
   - `f1grid`: the old F1 grid, only for ≤ 4 members.
4. **Nested 5-fold evaluation.** Calibration and weights are fitted on 4/5 of the OOF and scored on 1/5 in three scenarios:
   - `plain`;
   - `shift`: held-out rows resampled to the EM test class mix;
   - `shift_conf`: also matched to the test's lower confidence (the stress test from §A/§D).

   Deltas are paired against `equal`, with standard errors. The script also reports how much the weights vary across folds; large variation is a warning sign of over-fitting.
5. **Conservative selection.** A weighted method replaces `equal` only if its Δ ≥ 0 in **every** scenario and its plain Δ > 2 se. Among those, the one with the best worst-case Δ wins. Otherwise `equal` is kept.
6. **Refit on all OOF**, apply to the test, and write:
   - `t4_blend.csv` (`id,id_img,pred_num,pred_str`), `t4_blend_test_str.csv`, `t4_blend_test_str_711K.csv`;
   - test probabilities and the nested OOF of the selected method;
   - `blend_summary.json` and `BLEND_REPORT.md`.
7. **Label-free test diagnostics**, reported only and **never used for selection**: ATC accuracy per member and for the blend, disagreement between members on test, and fold disagreement per member.

Verification:
- **v3 alone:** selects `equal` and reproduces `t4_test_str.csv` exactly.
- **Self-test** (v3 + a differently-sharpened noisy copy of v3 + a clearly bad member, single OOF 0.54):

| method | nested OOF mF1 | Δ plain | Δ shift | Δ shift_conf | weights v3 / copy / bad |
|---|---|---|---|---|---|
| equal | 0.98827 | 0 | 0 | 0 | .33 / .33 / .33 |
| **nll (selected)** | 0.98879 | +0.00053 ± 0.00022 | +0.00042 | +0.00285 | .97 / .02 / .01 |
| nll_bag | 0.98879 | +0.00053 | +0.00042 | +0.00285 | .95 / .04 / .01 |
| caruana | 0.98876 | +0.00049 | +0.00038 | +0.00265 | .97 / .03 / .00 |
| shrink | 0.98849 | +0.00023 | +0.00021 | +0.00193 | .65 / .18 / .17 |
| f1grid | 0.98878 | +0.00051 | +0.00037 | +0.00309 | .98 / .00 / .02 |

  The bad member drags the equal blend down, most of all under `shift_conf`. Every weighted method removes it.
  The noisy copy adds no information (it is v3 plus noise), so it correctly gets ~0 weight.
  This only checks the selection logic. Real gains need real, *diverse* members (v4 EVA02 / ConvNeXt-S): with two members of similar quality, expect `equal` or `shrink` to win.
- **Runtime:** about 1 min for 3 members on CPU.

---

## Recommendation
0. **Expect T4 on the private LB to be ≈ 0.87–0.94, not 0.989 (§D1).** The test is shifted in image content, mainly freefried / fiveN / pube. That is why the v4 robustness run plus a second backbone is the most valuable T4 work left, more than any post-processing.
1. T4 final = **v3_5f plain argmax** (`t4_test_str.csv` strings, or v6's numeric column), with **no prior-shift correction and no logit bias**.
2. If GPU time allows, run `task4_kaggle_v4.py` for `eva02_s` and `cnxt_s_d3`. Blend with `t4_robust_blend.py` (§E). Use its selected blend if its nested OOF ≥ 0.9889 and its Δ vs v3 alone is not negative in any scenario.
3. For the string-vs-number question: the slide's sample submission uses strings in the task4 column.
   If the team decides the private LB expects strings, put `t4_test_str.csv` (or the 711K variant) into the task4 column of the final file. That changes only T4, which the public LB does not score.
