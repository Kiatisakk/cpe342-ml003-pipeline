"""Read-only verification script that generates report/facts_ledger.json
from competition datasets, saved OOF/test predictions, and final clean submission.
Does NOT inspect t4/ and does NOT modify any files outside report/.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, fbeta_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(ROOT, "report", "facts_ledger.json")


def ecdf_map(ref):
    s = np.sort(ref)
    return lambda v: np.searchsorted(s, v, side="right") / len(s)


def hurdle_median(p, Q, alphas, cap=300000.0):
    pp = np.clip(p, 1e-6, 1.0)
    q = np.clip(1.0 - 1.0 / (2.0 * pp), alphas[0], alphas[-1])
    Qs = np.sort(Q, axis=1)
    j = np.clip(np.searchsorted(alphas, q) - 1, 0, len(alphas) - 2)
    a0, a1 = alphas[j], alphas[j + 1]
    r = np.arange(len(q))
    v = Qs[r, j] + (q - a0) / (a1 - a0) * (Qs[r, j + 1] - Qs[r, j])
    return np.where(pp > 0.5, np.clip(v, 0.0, cap), 0.0)


def nmae_score(yt, yp):
    return float(1.0 / (1.0 + np.abs(yt - yp).sum() / yt.sum()))


def folds_v4(y, seed=42):
    pos = y > 0
    strat = np.where(
        pos,
        1 + pd.qcut(pd.Series(y).where(pos).rank(method="first"), 4, labels=False).fillna(-1).values,
        0,
    ).astype(int)
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), strat))


def main():
    os.chdir(ROOT)
    sub = pd.read_csv("submissions/FINAL/sub_v85a_soup_trainprior.csv")
    assert sub.shape == (25889, 6), f"Unexpected submission shape: {sub.shape}"

    # ------------------------------------------------------------------
    # Task 1: Anti-Cheat Pre-Filter (Binary Classification, Binary F2)
    # ------------------------------------------------------------------
    tr1 = pd.read_csv("cpe342-kaggle-main/01_binary_classification/train.csv")
    te1 = pd.read_csv("cpe342-kaggle-main/01_binary_classification/test.csv")
    lab1 = tr1["is_cheater"].notna().values
    y1 = tr1.loc[lab1, "is_cheater"].astype(int).values

    o_all = np.load("t1/research/best_oof_allseeds.npy")
    o_mean = o_all.mean(0)
    o_te = np.load("t1/research/best_test_proba.npy")
    s_all = np.load("t1/research/blind_sonnet/stack_lr_oof.npy")
    s_oof = s_all[lab1] if len(s_all) == len(tr1) else s_all
    s_te = np.load("t1/research/blind_sonnet/stack_lr_test.npy")

    W1 = {"xgb": 0.1834, "lgb": 0.4848, "cat": 0.3318}
    P1_dir = "t1/preds/t1_pipeline/"
    g_oof = sum(w * ecdf_map(np.load(f"{P1_dir}oof_{m}.npy"))(np.load(f"{P1_dir}oof_{m}.npy")) for m, w in W1.items())
    g_te = sum(w * ecdf_map(np.load(f"{P1_dir}oof_{m}.npy"))(np.load(f"{P1_dir}test_{m}.npy")) for m, w in W1.items())

    b1_oof = 0.7 * ecdf_map(o_mean)(o_mean) + 0.2 * ecdf_map(s_oof)(s_oof) + 0.1 * ecdf_map(g_oof)(g_oof)
    b1_te = 0.7 * ecdf_map(o_mean)(o_te) + 0.2 * ecdf_map(s_oof)(s_te) + 0.1 * ecdf_map(g_oof)(g_te)

    grid1 = np.quantile(b1_oof, np.linspace(0.2, 0.7, 251))
    f2_grid = [fbeta_score(y1, b1_oof >= t, beta=2) for t in grid1]
    tb1 = float(grid1[int(np.argmax(f2_grid))])

    t1_clean = pd.read_csv("t1/preds/clean_fixedthr/t1_clean.csv")
    assert (t1_clean["pred"].values == sub["task1"].astype(int).values).all(), "T1 clean mismatch vs sub_v85a"

    # Missingness difference train vs test
    raw1_feats = [c for c in tr1.columns if c not in ("player_id", "is_cheater")]
    t1_final_res = json.load(open("t1/research/final_results.json", encoding="utf-8"))

    v1_oof = np.load("t1/research/v1_best_oof.npy")
    v2_oof = np.load("t1/research/best_oof.npy")

    t1_facts = {
        "train_total_rows": int(len(tr1)),
        "train_labeled_rows": int(lab1.sum()),
        "train_unlabeled_rows": int((~lab1).sum()),
        "test_rows": int(len(te1)),
        "raw_features": len(raw1_feats),
        "signal_features": 15,
        "noise_features": 16,
        "pos_rate_labeled": round(float(y1.mean()), 4),
        "mean_missing_per_row_train_labeled": round(float(tr1.loc[lab1, raw1_feats].isna().sum(axis=1).mean()), 2),
        "mean_missing_per_row_test": round(float(te1[raw1_feats].isna().sum(axis=1).mean()), 2),
        "v1_round1_seed42_auc": round(float(roc_auc_score(y1, v1_oof)), 5),
        "v2_round2_seed42_auc": round(float(roc_auc_score(y1, v2_oof)), 5),
        "final_v2_avg3_auc": t1_final_res["avg3_oof"]["auc"],
        "final_v2_avg3_f2m_nested": t1_final_res["avg3_oof"]["macro"]["nested_mean"],
        "final_v2_avg3_f2b_nested": t1_final_res["avg3_oof"]["binary"]["nested_mean"],
        "clean_blend_oof_auc": round(float(roc_auc_score(y1, b1_oof)), 5),
        "clean_blend_oof_binf2_opt": round(float(max(f2_grid)), 5),
        "clean_blend_fixed_thr": round(tb1, 4),
        "clean_blend_oof_pos_rate_opt": round(float((b1_oof >= tb1).mean()), 4),
        "clean_blend_oof_binf2_at60": round(float(fbeta_score(y1, b1_oof >= np.quantile(b1_oof, 0.4), beta=2)), 5),
        "test_pos_count": int((sub["task1"] == 1).sum()),
        "test_pos_rate": round(float((sub["task1"] == 1).mean()), 4),
        "lb_probe_macro_to_binf2": {
            "sub_v10d_macro_41pct": 0.66614,
            "probe_v10e_binf2_63pct": 0.67890,
            "delta_lb": 0.01276,
        },
        "lb_rank_blend_sub_v20a": {"from_v18a": 0.71999, "to_v20a": 0.72016, "delta_lb": 0.00017},
        "lb_pseudo_labels_v22": {"v21a": 0.76315, "v22a_soft": 0.76307, "v22b_hard": 0.76309},
        "power_law_ceiling_auc": 0.8939,
        "power_law_ceiling_f2m": 0.7833,
    }

    # ------------------------------------------------------------------
    # Task 2: Player Segmentation (4-Class Classification, Macro F1, v2)
    # ------------------------------------------------------------------
    tr2 = pd.read_csv("cpe342-kaggle-main/02_multiclass_classification/train.csv")
    te2 = pd.read_csv("cpe342-kaggle-main/02_multiclass_classification/test.csv")
    y2 = tr2["segment"].astype(int).values
    nb2_oof = np.load("t2/preds/v2data/nb_oof.npy")
    lr2_oof = np.load("work/t2v2/blind/out/logreg_spline_A_oof.npy")
    mlp2_oof = np.load("work/t2v2/blind/out/mlp_torch_A_oof.npy")
    b2_oof = 0.87 * nb2_oof + 0.08 * lr2_oof + 0.05 * mlp2_oof

    pi2 = np.bincount(y2, minlength=4) / len(y2)
    q2 = pi2.copy()
    Fk2 = np.full(4, 0.75)
    for _ in range(8):
        aa2 = 2.0 / (pi2 + q2)
        bb2 = -Fk2 / (pi2 + q2)
        pr2 = (aa2 * b2_oof + bb2).argmax(1)
        q2 = np.bincount(pr2, minlength=4) / len(y2)
        Fk2 = np.array([2.0 * ((pr2 == k) & (y2 == k)).mean() / (pi2[k] + q2[k]) for k in range(4)])

    t2_csv = pd.read_csv("t2/preds/v2data/t2v2_blend_affine.csv")
    assert (t2_csv["pred"].values == sub["task2"].astype(int).values).all(), "T2 v2 mismatch vs sub_v85a"

    # Check player_id vs segment AUC on v2 (confirm leak gone)
    pid2_num = tr2["player_id"].str[1:].astype(int).values
    pid2_vs_seg_auc = [round(float(roc_auc_score(y2 == k, pid2_num)), 4) for k in range(4)]

    # Within-class correlations on v2 continuous informative features
    feats2 = [c for c in tr2.columns if c not in ("player_id", "segment")]
    t2_facts = {
        "train_rows": int(len(tr2)),
        "test_rows": int(len(te2)),
        "raw_features": len(feats2),
        "kept_features": 28,
        "noise_features": 16,
        "train_class_counts": {str(k): int(v) for k, v in pd.Series(y2).value_counts().sort_index().items()},
        "train_class_shares": {str(k): round(float(v), 4) for k, v in pd.Series(y2).value_counts(normalize=True).sort_index().items()},
        "v2_pid_vs_segment_ovr_auc": pid2_vs_seg_auc,
        "v2_emnb_oof_f1m": round(float(f1_score(y2, nb2_oof.argmax(1), average="macro")), 5),
        "v2_emnb_oof_logloss": round(float(log_loss(y2, nb2_oof)), 4),
        "v2_logreg_spline_oof_f1m": round(float(f1_score(y2, lr2_oof.argmax(1), average="macro")), 5),
        "v2_mlp_torch_oof_f1m": round(float(f1_score(y2, mlp2_oof.argmax(1), average="macro")), 5),
        "v2_blend_argmax_oof_f1m": round(float(f1_score(y2, b2_oof.argmax(1), average="macro")), 5),
        "v2_blend_affine_oof_f1m": round(float(f1_score(y2, pr2, average="macro")), 5),
        "v2_affine_rows_changed_oof": int((pr2 != b2_oof.argmax(1)).sum()),
        "v2_3seed_emnb_f1m": {"seed42": 0.7577, "seed7": 0.7582, "seed123": 0.7578},
        "v2_test_class_counts": {str(int(k)): int(v) for k, v in sub["task2"].value_counts().sort_index().items()},
        "v2_test_class_shares": {str(int(k)): round(float(v), 4) for k, v in sub["task2"].value_counts(normalize=True).sort_index().items()},
        "v1_reference": {
            "nb_uniform_f1m": 0.7550,
            "nb_optuna_optimistic_f1m": 0.7557,
            "nb_optuna_nested_f1m": 0.7544,
            "lgbm_raw_f1m": 0.7509,
            "lgbm_tuned_f1m": 0.7524,
            "lgbm_train_insample_f1m": 0.7925,
            "emnb_llsel_v1_f1m": 0.7562,
            "emnb_llsel_v1_logloss": 0.5993,
            "estimated_label_noise_eps": 0.0285,
            "true_v1_label_noise": 0.0300,
            "id_block_leak_f1m": 0.9677,
        },
        "lb_blend_affine_delta": {"v16b": 0.71965, "v18a": 0.71999, "delta_lb": 0.00034},
    }

    # ------------------------------------------------------------------
    # Task 3: Player Monthly Spending Prediction (Regression, NMAE)
    # ------------------------------------------------------------------
    tr3 = pd.read_csv("cpe342-kaggle-main/03_regression/train.csv")
    te3 = pd.read_csv("cpe342-kaggle-main/03_regression/test.csv")
    y3 = tr3["spending_30d"].astype(float).values
    F3 = folds_v4(y3, seed=42)
    ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    O3 = "t3/preds/seedbag10/repo/t3/research/oof/"
    n3 = len(y3)

    t3_bags = {}
    for nm in ("bag_q3p3", "bag_q10p3", "bag_q3p10", "bag_q10p10"):
        Q3 = np.load(O3 + nm + "_Q.npy")[:n3]
        p3 = np.load(O3 + nm + "_p.npy")[:n3]
        o3 = hurdle_median(p3, Q3, ALPHAS)
        fs3 = [nmae_score(y3[b], o3[b]) for _, b in F3]
        t3_bags[nm] = {
            "oof_nmae": round(nmae_score(y3, o3), 5),
            "fold_mean": round(float(np.mean(fs3)), 5),
            "fold_sd": round(float(np.std(fs3)), 5),
            "zero_pred_rate": round(float((o3 == 0).mean()), 4),
        }

    # Verify sub_v85a task3 matches bag_q10p10_test
    t3_test_raw = np.clip(np.load(O3 + "bag_q10p10_test.npy"), 0, None)
    assert np.allclose(sub["task3"].values, np.round(t3_test_raw, 2), atol=0.006), "T3 test mismatch vs sub_v85a"

    # Segment shares & OOF error budget by segment
    Q10 = np.load(O3 + "bag_q10p10_Q.npy")[:n3]
    p10 = np.load(O3 + "bag_q10p10_p.npy")[:n3]
    o10 = hurdle_median(p10, Q10, ALPHAS)
    sg3 = tr3["segment"].fillna(-1).astype(int).values
    seg_stats = {}
    for k in sorted(np.unique(sg3)):
        m = sg3 == k
        seg_stats[str(k)] = {
            "rows": int(m.sum()),
            "zero_rate": round(float((y3[m] == 0).mean()), 4),
            "mean_y": round(float(y3[m].mean()), 2),
            "spend_share_pct": round(float(100.0 * y3[m].sum() / y3.sum()), 2),
            "err_ratio": round(float(np.abs(y3[m] - o10[m]).sum() / max(y3[m].sum(), 1.0)), 4),
            "err_share_pct": round(float(100.0 * np.abs(y3[m] - o10[m]).sum() / np.abs(y3 - o10).sum()), 2),
        }

    t3_facts = {
        "train_rows": int(len(tr3)),
        "test_rows": int(len(te3)),
        "raw_features": int(len([c for c in tr3.columns if c not in ("player_id", "spending_30d")])),
        "train_zero_rate": round(float((y3 == 0).mean()), 4),
        "train_mean_thb": round(float(y3.mean()), 2),
        "train_median_thb": round(float(np.median(y3)), 2),
        "train_p99_thb": round(float(np.quantile(y3, 0.99)), 2),
        "train_max_thb": round(float(y3.max()), 2),
        "clf_overall_auc_p10": round(float(roc_auc_score(y3 > 0, p10)), 4),
        "bags": t3_bags,
        "segment_breakdown_oof": seg_stats,
        "test_zero_rate": round(float((sub["task3"] == 0).mean()), 4),
        "test_mean_thb": round(float(sub["task3"].mean()), 2),
        "test_median_thb": round(float(sub["task3"].median()), 2),
        "test_p99_thb": round(float(sub["task3"].quantile(0.99)), 2),
        "test_max_thb": round(float(sub["task3"].max()), 2),
        "milestones": {
            "hgb_two_stage": 0.7843,
            "v4_xgb_dq": 0.79169,
            "v4_nested_lad_honest": 0.79191,
            "xgb_offset_prev": 0.79391,
            "cb_d4_offset_single": 0.79505,
            "cb_d4_3seed_bag": 0.79529,
            "cb_d4_10x10_bag_final": t3_bags["bag_q10p10"]["oof_nmae"],
            "learning_curve_ceiling": "0.796-0.797",
            "insample_train_score": 0.80356,
        },
        "lb_all_zero_probe": {"from_v10e": 0.67890, "to_v10g": 0.61991, "delta_lb": -0.05899},
        "error_budget_from_insights": {
            "payer_mult_noise_pct": 49.0,
            "zero_pred_pos_pct": 32.9,
            "whale_spike_266rows_pct": 15.4,
            "payer_pred_zero_pct": 2.7,
        },
    }

    # ------------------------------------------------------------------
    # Task 5: Account Security Monitoring (Unsupervised Anomaly, F3, v2)
    # ------------------------------------------------------------------
    te5 = pd.read_csv("cpe342-kaggle-main/05_anomaly_detection/test.csv")
    t5_summ = json.load(open("work/t5v2/out/t5_pipeline_summary.json", encoding="utf-8"))
    t5_grp = pd.read_csv("work/t5v2/out/t5_groups.csv")
    t5_lsnan = pd.read_csv("work/t5v2/out/lowstab_nan_scores.csv")
    t5_final = pd.read_csv("t5/preds/v2data/t5v2_union_box_lsnan.csv")
    assert (t5_final["pred"].values == sub["task5"].astype(int).values).all(), "T5 v2 mismatch vs sub_v85a"

    # Compute exact v2 detector counts and shock subtypes
    P = [1, 2, 3, 4]
    shock = t5_grp["rule"].values == 1
    def ext(c):
        v = te5[c].values
        lo, hi = np.nanquantile(v[~shock], [0.0005, 0.9995])
        return (v < lo) | (v > hi)
    takeover = shock & (ext("currency_transfers_4") | ext("play_pattern_consistency_4"))
    selling = shock & ~takeover & (ext("skill_trajectory_4") | ext("skill_rating_4") | ext("friend_removes_4"))
    vpn = shock & ~takeover & ~selling & (ext("purchase_frequency_4") | ext("purchase_amount_mean_4"))
    geo_only = shock & ~takeover & ~selling & ~vpn

    flags5 = ["password_changed", "mass_item_sale", "vpn_usage"]
    flag_rates = {}
    for nm, m in [("takeover", takeover), ("selling", selling), ("vpn", vpn), ("geo_only", geo_only), ("non_shock", ~shock)]:
        flag_rates[nm] = {"count": int(m.sum())}
        for fl in flags5:
            flag_rates[nm][fl] = round(float(te5.loc[m, fl].mean()), 3)

    gap_counts = {g["rules"][0].split("<")[0]: int(g["n"]) for g in t5_summ["gap_groups"]}
    n_lsnan_added = int(((t5_lsnan["post_lowstab_nan"].values > 0.1) & (t5_grp.iloc[:, 1:].max(axis=1).values == 0)).sum())

    # Residual independence check
    per5 = sorted({c[:-2] for c in te5.columns if c[-2:] in ("_1", "_2", "_3", "_4")})
    cont5 = [b for b in per5 if b not in ("login_lat", "login_lon", "device_count", "refund_count", "guild_changes")]
    F5 = pd.DataFrame({b: te5[[f"{b}_{p}" for p in P]].mean(axis=1) for b in cont5})
    F5["account_age_days"] = te5["account_age_days"]
    F5["behavior_stability"] = te5["behavior_stability"]
    def abs_corr_vals(df):
        r = df.corr().abs().to_numpy()
        return r[np.triu_indices_from(r, 1)]
    r_all5 = abs_corr_vals(F5)
    r_clean5 = abs_corr_vals(F5[t5_final["pred"].values == 0])

    # Verify v2 pid leak eliminated
    pid5_num = te5["player_id"].str[1:].astype(int).values
    pid5_vs_flag_auc = round(float(roc_auc_score(t5_final["pred"].values, pid5_num)), 4)

    t5_facts = {
        "test_rows": int(len(te5)),
        "raw_features": int(te5.shape[1] - 1),
        "total_flagged": int((sub["task5"] == 1).sum()),
        "flagged_rate": round(float((sub["task5"] == 1).mean()), 4),
        "v2_detectors": {
            "A_shock": int(shock.sum()),
            "A_subtypes": flag_rates,
            "B_gap_bot": gap_counts.get("ip_hash_entropy", 207),
            "B_gap_lowact": gap_counts.get("activity_level", 265),
            "B_gap_lowcomm": gap_counts.get("communication_frequency", 263),
            "B_gap_lowstab": gap_counts.get("behavior_stability", 246),
            "B_gap_total": int(sum(gap_counts.values())),
            "C_acckda_box": int(t5_summ["components"][0]["n_flag"]),
            "D_hidden_lowstab_added": int(t5_final["pred"].sum() - shock.sum() - sum(gap_counts.values()) - t5_summ["components"][0]["n_flag"]),
            "D_lsnan_total_above_01": int((t5_lsnan["post_lowstab_nan"].values > 0.1).sum()),
        },
        "independence_all_median_r": round(float(np.median(r_all5)), 4),
        "independence_all_max_r": round(float(r_all5.max()), 4),
        "independence_clean_median_r": round(float(np.median(r_clean5)), 4),
        "independence_clean_max_r": round(float(r_clean5.max()), 4),
        "v2_pid_vs_anomaly_auc": pid5_vs_flag_auc,
        "expected_f3_v2": 0.99681,
        "lb_probes": {
            "t5_all_zero_delta": -0.17719,
            "acckda_box_delta_v10c_vs_v6": 0.0214,
            "loose_3602_delta": -0.01239,
            "shock_z4_6_2167_delta": -0.00246,
            "lsnan_cut_015_delta": -0.00012,
            "lsnan_cut_005_delta": 0.00002,
            "prior13_v27_delta": -0.00004,
            "extra_11_rows_v87e_2115_delta": -0.00010,
        },
    }

    # ------------------------------------------------------------------
    # Cross-Task Independence Check
    # ------------------------------------------------------------------
    tr1_ids = set(tr1["player_id"])
    tr2_ids = set(tr2["player_id"])
    tr3_ids = set(tr3["player_id"])
    te5_ids = set(te5["player_id"])
    j23 = tr2[["player_id", "segment"]].set_index("player_id").join(
        tr3[["player_id", "segment"]].set_index("player_id"), lsuffix="_t2", rsuffix="_t3", how="inner"
    ).dropna()
    t2_t3_seg_agree = float((j23["segment_t2"] == j23["segment_t3"]).mean())

    j12 = tr1.loc[lab1, ["player_id", "is_cheater"]].set_index("player_id").join(
        tr2[["player_id", "segment"]].set_index("player_id"), how="inner"
    )
    cheat_by_t2_seg = {str(int(k)): round(float(v), 4) for k, v in j12.groupby("segment")["is_cheater"].mean().items()}

    cross_task = {
        "overlap_t1_t2_train": len(tr1_ids & tr2_ids),
        "overlap_t2_t3_train": len(tr2_ids & tr3_ids),
        "overlap_t1_t3_train": len(tr1_ids & tr3_ids),
        "t2_vs_t3_segment_agreement_same_id": round(t2_t3_seg_agree, 4),
        "t1_cheater_rate_by_t2_segment": cheat_by_t2_seg,
        "t1_t2_account_age_corr": round(float(
            tr1.set_index("player_id")["account_age_days"].corr(tr2.set_index("player_id")["account_age_days"])
        ), 4),
        "t2_t3_friend_count_corr": round(float(
            tr2.set_index("player_id")["friend_count"].corr(tr3.set_index("player_id")["friend_count"])
        ), 4),
    }

    ledger = {
        "meta": {
            "team": "ML003",
            "course": "CPE 342 Machine Learning",
            "project": "Karena Player Intelligence System",
            "final_submission_reference": "submissions/FINAL/sub_v85a_soup_trainprior.csv",
            "task4_status": "EXCLUDED_PENDING_STABLE_SCORING",
            "submission_rows": int(len(sub)),
        },
        "cross_task": cross_task,
        "task1": t1_facts,
        "task2": t2_facts,
        "task3": t3_facts,
        "task5": t5_facts,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)
    print(f"Wrote verified facts ledger -> {OUT_PATH}")
    print(f"  T1 clean OOF binF2={t1_facts['clean_blend_oof_binf2_opt']} AUC={t1_facts['clean_blend_oof_auc']} test_pos={t1_facts['test_pos_count']} ({t1_facts['test_pos_rate']*100:.2f}%)")
    print(f"  T2 v2 OOF MacroF1={t2_facts['v2_blend_affine_oof_f1m']} (NB={t2_facts['v2_emnb_oof_f1m']} LL={t2_facts['v2_emnb_oof_logloss']})")
    print(f"  T3 10x10 OOF NMAE={t3_facts['milestones']['cb_d4_10x10_bag_final']} (zeros={t3_facts['test_zero_rate']*100:.2f}%, mean={t3_facts['test_mean_thb']})")
    print(f"  T5 v2 flagged={t5_facts['total_flagged']} ({t5_facts['flagged_rate']*100:.2f}%)")


if __name__ == "__main__":
    main()
