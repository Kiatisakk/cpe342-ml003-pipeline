# T1 round 9 (sonnet): generator-artefact hunt

Nothing submitted. All scripts here (eda1-3.py, exp_mask.py).

## Artefacts checked (labelled train, 90,816 rows)
- player_id: unique, no train/test overlap, no duplicate feature rows. Label rate vs id mod 2..1000 and blocks of 10/50/100/500: spread equals binomial noise. LightGBM on id + mods: AUC 0.4995. Train row order autocorrelation -8e-5. Unlabeled share is flat across id. Adversarial AUC train/test on id alone: 0.500.
- Missingness: MCAR in train. Label rate flat vs per-column missing and vs number missing. No MNAR.
- Decimals/precision: float noise only. Exact repeated values are clip bounds (e.g. kdr 12.0, headshot 5/95, win_rate 100, rt 55, aiming/spray/game_sense 1.0); trees already see them.
- Semi-supervised/pseudo-label: already negative (exp07, exp11, exp19); no new signal found, not retried.

## One real finding: test missingness differs from train
Adversarial AUC train vs test on features = 0.90, driven only by per-column missing rates (KS on values <= 0.011).
Examples (train -> test): account_age 14.4% -> 0.6%, friend_network 22% -> 4.7%, aiming 21% -> 2.1%, level 7.1% -> 1.5%; headshot 2.7% -> 14.1%, win_rate 5% -> 13.7%, game_sense 2.7% -> 6.7%.
Mean missing cells per row 3.7 -> 2.5. Test is more informative (explains LB 0.849 > CV 0.839). Not a label leak.

Consequences
1. Threshold: the current binary file cuts at a fixed 62.96% positive rate (~prob 0.145 on test). Calibrated-prob cut is shift-robust; OOF optimum prob cut = 0.19 (F2 0.8395 vs 0.8372 at 0.145). On test, prob 0.18 -> 60.6% positive. Expected gain ~ +0.001-0.002 (flat curve).
   File: t1_probthr018.csv (id,pred,score; same scores as t1_best_test.csv).
2. Test-shift mask augmentation (exp_mask.py; extra masking only on columns where test misses more: headshot, win_rate, accuracy, game_sense, kill_consistency), LGB fe_v2, folds 0-1 of seed 42, validation masked test-like:
   N (no aug) F2b 0.83808 / AUC 0.89263; B (test-shift + train-rate aug) 0.83857 / 0.89294. +0.0005 F2: below the 0.003 bar, noise-level. Not adopted.

## Verdict
No leak/generator artefact. No CV gain > 0.003. Only optional tweak: prob threshold ~0.18-0.19 instead of rate 63%.
