# T2 blind report

Folds: StratifiedKFold(5,shuffle,42). Macro F1 mean/std/runtime(s). Machine contended, runtimes inflated.

| experiment | F1 | std | s | verdict |
|--|--|--|--|--|
| lgb_all_A | 0.7488 | 0.0035 | 306 | |
| lgb_all_B | 0.9661 | 0.0008 | 399 | |
| nb_hist_20_1.0_A | 0.7511 | 0.0031 | 7 | |
| nb_hist_40_1.5_A | 0.7489 | 0.0035 | 10 | |
| svm_linear_A | 0.7304 | 0.0043 | 45 | |
| nb_12_1.0_A | 0.7494 | 0.0033 | 7 | |
| nb_20_1.0_A | 0.7511 | 0.0031 | 5 | |
| nb_20_2.0_A | 0.7467 | 0.0029 | 5 | |
| nb_30_2.0_A | 0.7475 | 0.0032 | 5 | |
| nb_30_3.0_A | 0.7442 | 0.0028 | 6 | |
| nb_idcleanlabels_20_1.0_A* | 0.7515 | 0.0034 | 9 | |
| nb_idcleanlabels_30_1.0_A* | 0.7523 | 0.0038 | 9 | |
| nb_idcleanlabels_30_1.0_Aclean | 0.7523 | 0.0038 | 8 | |
| nb_idcleanlabels_40_1.0_Aclean | 0.7516 | 0.0040 | 8 | |
| svm_nystroem300_g.01_A | 0.7406 | 0.0043 | 121 | |
| svm_nystroem300_g.003_A | 0.7389 | 0.0043 | 114 | |
| svm_rbf_sub8k_g.01_A | 0.7395 | 0.0046 | 365 | |
| lda_A | 0.7336 | 0.0043 | 17 | |
| qda_A | 0.6757 | 0.0036 | 10 | |
| logreg_spline_A | 0.7521 | 0.0033 | 379 | |
| extratrees_A | 0.7057 | 0.0020 | 249 | |
| mlp_torch_A | 0.7511 | 0.0030 | 129 | |

Not run / skipped: kNN (script in fam.py, not executed for time), RF (dropped), stacking (no family beat NB by >0.001, so skipped), nested class-bias tuning.
Verdicts: Track A ceiling ~0.752: all good families (NB, spline logreg, MLP, LGBM) land at 0.749-0.752 within noise. Weaker: linear SVM 0.730, Nystroem/RBF SVM 0.739-0.741 (subsampled/300 comps), LDA 0.734, QDA 0.676, ExtraTrees 0.706.
Track B: id rule  (pid<52000->0, <84500->1, <110500->2, else 3) gives 0.968; LGBM with pid 0.966. 
## Artefacts
- player_id numeric part is sorted by TRUE segment (blocks 0-52k,52k-84.5k,84.5k-110.5k,110.5k+). ~3.0% of labels disagree with the block, and those flips are unrelated to features (NB predicts the id-block class for them), i.e. label noise; Track B ceiling ~0.968.
- Within-class feature correlations <0.03: conditional independence, so NB is near Bayes-optimal; noise cols: random_metric_*, preferred_game_mode, account_age_days, avg_match_length, responds_to_discounts, item_type_preference_social. Informative categoricals: engagement_level, loyalty_tier only.
- nb_idcleanlabels_30_Aclean trains on id-block labels (id used in training only, test inference id-free): 0.7523, +0.001 over plain NB; flag as hybrid.
Saved oof/test: logreg_spline_A, nb_idcleanlabels_30_1.0_Aclean, mlp_torch_A, nb_hist_40_1.5_A, lgb_all_A, extratrees_A, all svm_*; B: idrule_B, lgb_all_B.
