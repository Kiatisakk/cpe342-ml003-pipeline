# T1 blind re-derivation (is_cheater, F2 of positive class)
Validation: StratifiedKFold(5, shuffle, rs=42) on 90,816 labeled rows; threshold chosen per fold by inner CV (3-fold; 2 for slower models) on the training part only. F2 = mean over folds, std across folds. Runtimes inflated by a heavily shared CPU.
Reference: always-predict-positive F2 ~0.729.

| experiment | F2 | std | AUC | sec | verdict |
|---|---|---|---|---|---|
| LightGBM, all 31 raw features | 0.8370 | 0.0011 | 0.8915 | 277 | noise columns add nothing |
| LightGBM, 15 signal features | 0.8374 | 0.0016 | 0.8920 | 372 | = |
| LightGBM, 15 signal + engineered (age/level bounds from level_progression_speed identity, ratios) | 0.8374 | 0.0010 | 0.8928 | 291 | best; saved (gbm_eng_sig) |
| Linear SVM (LinearSVC, quantile-free std scaling, NaN flags) | 0.7651 | 0.0016 | 0.7663 | 60 | weak (near trivial) |
| Linear SVM on spline basis (8 knots) | 0.8058 | 0.0021 | 0.8391 | 294 | best SVM |
| Nystroem(300, gamma .02) + SGD hinge | 0.7561 | 0.0054 | 0.7698 | 236 | poor, untuned |
| RBF SVC, 10k subsample (C=1, gamma=scale) | 0.7556 | 0.0011 | 0.7646 | 500 | weak (subsample + approx features limit it) |
| Gaussian NB (quantile-normal) | 0.7513 | 0.0021 | 0.7304 | 74 | weak |
| ExtraTrees 200, leaf 10, NaN=-1 | 0.8110 | 0.0021 | 0.8611 | 425 | below GBM |
| kNN k=100, 20k subsample, quantile-normal | 0.7284 | 0.0042 | 0.6893 | 143 | no better than trivial |
| MLP 128-64, 12 epochs, torch | 0.8051 | 0.0022 | 0.8468 | 125 | below GBM |
| Stack: logistic on logits of gbm, et, mlp, svm_spline (second level on OOFs, same folds) | 0.8383 | 0.0010 | 0.8928 | 5 | +0.001 over GBM, within noise; weights gbm 1.23, svm -0.51 |

Saved (also et, knn, nn_mlp, svm_rbf_sub10k, stack_lr): gbm_eng_sig, svm_linear, svm_spline_linear, svm_nystroem_g02, nb (each _oof.npy / _test.npy).

## Artefacts found
- 16 features are pure noise (AUC 0.5, equal distributions, no missingness signal). Missingness is uninformative.
- Class-exclusive hard ranges: reports_received legit 5-29 vs cheater 4-19 (cheaters 16-19 are 4x rarer; value 4 is cheater-only); cheater account_age<=1000 (legit to 1500), friends<=269 (legit 349), level<=96 (legit 100).
- level_progression_speed = level * r / account_age_days with r ~ U(0.15,1.08), same in both classes; gives bounds on a missing age/level.
- Row order / player_id carry no signal.
