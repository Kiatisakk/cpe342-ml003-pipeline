"""Generate publication-grade figures for Task 1 (Anti-Cheat Pre-Filter).

Outputs (under report/figures/t1/):
  1. fig_t1_signal_and_zones.png
  2. fig_t1_lps_identity.png
  3. fig_t1_f2_operating_point.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from plot_style import (  # noqa: E402
    AMBER,
    BLUE,
    CORAL,
    GRAY,
    LIGHT_GRAY,
    NAVY,
    SLATE,
    TEAL,
    apply_style,
)

OUT_DIR = REPO_ROOT / "report" / "figures" / "t1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_TRAIN = REPO_ROOT / "cpe342-kaggle-main" / "01_binary_classification" / "train.csv"
FACTS_PATH = REPO_ROOT / "report" / "facts_ledger.json"

SIGNAL_FEATURES = [
    "kill_death_ratio",
    "headshot_percentage",
    "win_rate",
    "accuracy_score",
    "kill_consistency",
    "reaction_time_ms",
    "account_age_days",
    "level",
    "level_progression_speed",
    "friend_network_size",
    "reports_received",
    "aiming_smoothness",
    "spray_control_score",
    "game_sense_score",
    "crosshair_placement",
]


def ecdf_transform(ref: np.ndarray):
    """Map values to empirical cumulative probability in `ref` (train OOF only)."""
    s = np.sort(ref)
    return lambda v: np.searchsorted(s, v, side="right") / len(s)


def load_labeled_train():
    """Load labeled train rows in file order, plus boolean mask over train.csv."""
    tr = pd.read_csv(DATA_TRAIN)
    lab_mask = tr["is_cheater"].notna().values
    lab = tr.loc[lab_mask].reset_index(drop=True)
    y = lab["is_cheater"].astype(int).values
    raw_cols = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
    return tr, lab, lab_mask, y, raw_cols


def make_fig1_signal_and_zones(lab: pd.DataFrame, y: np.ndarray, raw_cols: list[str], facts: dict):
    """Figure 1: Univariate |AUC - 0.5| separation + pure class-exclusive age zones."""
    t1_facts = facts["task1"]

    # Panel A: Univariate |AUC - 0.5| across all 31 raw features
    rows = []
    for col in raw_cols:
        mask = lab[col].notna().values
        auc = roc_auc_score(y[mask], lab.loc[mask, col].values)
        rows.append((col, abs(auc - 0.5), col in SIGNAL_FEATURES))
    df_auc = pd.DataFrame(rows, columns=["feature", "sep", "is_signal"]).sort_values(
        "sep", ascending=True
    ).reset_index(drop=True)

    min_sig = df_auc.loc[df_auc["is_signal"], "sep"].min()
    max_noise = df_auc.loc[~df_auc["is_signal"], "sep"].max()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": [1.05, 1.15]}
    )

    colors = [NAVY if s else GRAY for s in df_auc["is_signal"]]
    y_pos = np.arange(len(df_auc))
    ax1.barh(y_pos, df_auc["sep"], color=colors, height=0.72, edgecolor="none")
    ax1.axvspan(max_noise, min_sig, color=AMBER, alpha=0.18, zorder=0)
    ax1.axvline(0.5 * (max_noise + min_sig), color=CORAL, ls="--", lw=1.2, zorder=3)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_auc["feature"], fontsize=6.7)
    ax1.set_ylim(-0.7, len(df_auc) - 0.3)
    ax1.set_xlim(0, 0.222)
    ax1.grid(axis="y", visible=False)
    ax1.set_xlabel("Univariate |ROC-AUC - 0.5| (labeled train, n = 90,816)")
    ax1.set_title("(a) Signal vs. Pure Noise Across 31 Raw Features", pad=8)

    # Top-2 values subtle label for readability
    for idx in (len(df_auc) - 1, len(df_auc) - 2):
        val = df_auc.loc[idx, "sep"]
        ax1.text(val + 0.003, idx, f"{val:.4f}", va="center", ha="left", fontsize=7.2, color=NAVY)

    # Legend + clean separation callout in open lower-right area
    sig_patch = mpatches.Patch(color=NAVY, label="15 signal features")
    noise_patch = mpatches.Patch(color=GRAY, label="16 noise features")
    gap_patch = mpatches.Patch(color=AMBER, alpha=0.35, label="Separation gap")
    ax1.legend(
        handles=[sig_patch, noise_patch, gap_patch],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.24),
        fontsize=8.0,
    )
    ax1.text(
        0.96,
        0.05,
        f"Clean separation threshold:\n"
        f"min signal = {min_sig:.4f} (lps)\n"
        f"max noise = {max_noise:.4f} ({min_sig / max_noise:.1f}x gap)",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFBEB", edgecolor=AMBER, lw=0.9),
    )

    # Panel B: Empirical cheater rate binned by account_age_days (5-day bins up to 1300)
    # 5-day bins align cleanly with exact boundaries 15, 110, 120, 180, 200, 1000.
    age_valid = lab["account_age_days"].notna().values
    ag = lab.loc[age_valid, "account_age_days"].values
    yg = y[age_valid]

    edges = np.arange(0, 1305, 5)
    rates = np.array([yg[(ag >= a) & (ag < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    step_y = np.r_[rates, rates[-1]]

    ax2.axvspan(0, 15, color=TEAL, alpha=0.22, label="Pure legit zones (0% cheater)")
    ax2.axvspan(110, 120, color=CORAL, alpha=0.26, label="Pure cheater zone (100% cheater)")
    ax2.axvspan(180, 200, color=TEAL, alpha=0.22)
    ax2.axvspan(1000, 1300, color=TEAL, alpha=0.14)

    ax2.step(edges, step_y, where="post", color=NAVY, lw=1.35, label="Empirical cheater rate (5-day bins)")
    ax2.fill_between(edges, step_y, step="post", color=NAVY, alpha=0.07)

    base_rate = t1_facts["pos_rate_labeled"]
    ax2.axhline(base_rate, color=SLATE, ls="--", lw=1.2, label=f"Labeled base rate ({base_rate:.4f})")

    # Annotations placed in verified open regions
    n_lt15 = int((ag < 15).sum())
    n_110_120 = int(((ag >= 110) & (ag < 120)).sum())
    n_180_200 = int(((ag >= 180) & (ag < 200)).sum())
    n_ge1000 = int((ag >= 1000).sum())

    ax2.annotate(
        f"<15d (0%)\nn={n_lt15:,}",
        xy=(7.5, 0.015),
        xytext=(21, 0.13),
        ha="left",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color="#0F766E",
        arrowprops=dict(arrowstyle="->", color="#0F766E", lw=1.0),
    )
    ax2.annotate(
        f"110-120d\n(100%, n={n_110_120:,})",
        xy=(115, 0.99),
        xytext=(210, 0.91),
        ha="left",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color=CORAL,
        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.0),
    )
    ax2.annotate(
        f"180-200d\n(0%, n={n_180_200:,})",
        xy=(190, 0.015),
        xytext=(275, 0.13),
        ha="left",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color="#0F766E",
        arrowprops=dict(arrowstyle="->", color="#0F766E", lw=1.0),
    )
    ax2.text(
        1150,
        0.16,
        f">=1000d\n(0%, n={n_ge1000:,})",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color="#0F766E",
    )

    ax2.set_xlim(0, 1300)
    ax2.set_ylim(-0.03, 1.08)
    ax2.set_xlabel("account_age_days")
    ax2.set_ylabel("Empirical cheater rate P(is_cheater = 1)")
    ax2.set_title("(b) Pure Class-Exclusive account_age_days Windows", pad=8)
    ax2.legend(loc="upper right", fontsize=7.8)

    fig.tight_layout(w_pad=2.2)
    out_path = OUT_DIR / "fig_t1_signal_and_zones.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_fig2_lps_identity(lab: pd.DataFrame, y: np.ndarray, facts: dict):
    """Figure 2: Uniform r = lps * age / level identity + MCAR masking gain by missingness."""
    t1_facts = facts["task1"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.1))

    # Panel A: r = level_progression_speed * account_age_days / level
    r = (
        lab["level_progression_speed"] * lab["account_age_days"] / lab["level"]
    ).values
    mr = ~np.isnan(r)
    r_valid = r[mr]
    y_valid = y[mr]
    auc_r = roc_auc_score(y_valid, r_valid)

    bins = np.linspace(0.15, 1.08, 38)
    ax1.hist(
        r_valid[y_valid == 0],
        bins=bins,
        density=True,
        histtype="step",
        color=NAVY,
        lw=2.0,
        label=f"Legit (y = 0, n = {(y_valid == 0).sum():,})",
    )
    ax1.hist(
        r_valid[y_valid == 1],
        bins=bins,
        density=True,
        histtype="step",
        color=CORAL,
        lw=2.0,
        ls="--",
        label=f"Cheater (y = 1, n = {(y_valid == 1).sum():,})",
    )
    unif_density = 1.0 / (1.08 - 0.15)
    ax1.hlines(
        unif_density,
        0.15,
        1.08,
        colors=TEAL,
        linestyles=":",
        lw=1.6,
        label=f"Uniform(0.15, 1.08) density = {unif_density:.3f}",
    )

    ax1.set_xlim(0.08, 1.15)
    ax1.set_ylim(0.0, 1.46)
    ax1.set_xlabel("r = level_progression_speed * account_age_days / level")
    ax1.set_ylabel("Probability density")
    ax1.set_title("(a) Class-Invariant Progression Identity", pad=8)

    ax1.text(
        0.50,
        0.95,
        f"Univariate AUC = {auc_r:.4f} (no class leak)\n"
        f"Exact bounds when missing: age in [0.15, 1.08] * level / lps",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=7.9,
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.9),
    )
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 0.04), fontsize=8.0)

    # Panel B: OOF AUC stratified by missing signal feature count
    v1_oof = np.load(REPO_ROOT / "t1" / "research" / "v1_best_oof.npy")
    v2_oof = np.load(REPO_ROOT / "t1" / "research" / "best_oof.npy")
    n_miss = lab[SIGNAL_FEATURES].isna().sum(axis=1).values

    buckets = [
        ("0", n_miss == 0),
        ("1", n_miss == 1),
        ("2", n_miss == 2),
        ("3", n_miss == 3),
        ("4+", n_miss >= 4),
    ]
    auc_v1, auc_v2, counts, x_labels = [], [], [], []
    for lbl, mask in buckets:
        a1 = roc_auc_score(y[mask], v1_oof[mask])
        a2 = roc_auc_score(y[mask], v2_oof[mask])
        auc_v1.append(a1)
        auc_v2.append(a2)
        counts.append(int(mask.sum()))
        x_labels.append(f"{lbl}\n(n={mask.sum():,})")

    x = np.arange(len(buckets))
    w = 0.36
    b1 = ax2.bar(
        x - w / 2,
        auc_v1,
        width=w,
        color=GRAY,
        label=f"v1: No masking aug (OOF AUC {t1_facts['v1_round1_seed42_auc']:.5f})",
    )
    b2 = ax2.bar(
        x + w / 2,
        auc_v2,
        width=w,
        color=TEAL,
        label=f"v2: MCAR 4x masking aug (OOF AUC {t1_facts['v2_round2_seed42_auc']:.5f})",
    )

    for i, (a1, a2) in enumerate(zip(auc_v1, auc_v2)):
        gain = a2 - a1
        top = max(a1, a2)
        color = CORAL if i == len(buckets) - 1 else NAVY
        weight = "bold" if i >= 3 else "normal"
        ax2.text(
            x[i],
            top + 0.0025,
            f"+{gain:.5f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight=weight,
            color=color,
        )

    ax2.annotate(
        "38x larger gain\non 4+ missing",
        xy=(4 + w / 2, auc_v2[-1] + 0.011),
        xytext=(3.18, 0.885),
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=CORAL,
        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.0),
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=8.1)
    ax2.set_ylim(0.81, 0.945)
    ax2.set_xlabel("Number of missing signal features per row")
    ax2.set_ylabel("OOF ROC-AUC (seed 42; y-axis starts at 0.81)")
    ax2.set_title("(b) MCAR 4x Masking Gain by Missingness Depth", pad=8)
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="upper right", fontsize=7.8)

    fig.tight_layout(w_pad=2.2)
    out_path = OUT_DIR / "fig_t1_lps_identity.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_fig3_f2_operating_point(tr: pd.DataFrame, lab_mask: np.ndarray, y: np.ndarray, facts: dict):
    """Figure 3: Binary F2 vs Macro F2 operating curve on the final clean ECDF blend."""
    t1_facts = facts["task1"]

    # Reconstruct exact clean ECDF blend OOF (t1/code/t1_clean_fixedthr.py)
    v2_3seed_oof = np.load(REPO_ROOT / "t1" / "research" / "best_oof_allseeds.npy").mean(0)
    o_oof = ecdf_transform(v2_3seed_oof)(v2_3seed_oof)

    s_all = np.load(REPO_ROOT / "t1" / "research" / "blind_sonnet" / "stack_lr_oof.npy")
    s_ref = s_all[lab_mask] if len(s_all) == len(tr) else s_all
    s_oof = ecdf_transform(s_ref)(s_ref)

    weights = {"xgb": 0.1834, "lgb": 0.4848, "cat": 0.3318}
    pipe_dir = REPO_ROOT / "t1" / "preds" / "t1_pipeline"
    g_oof = np.zeros(len(y))
    for model_name, w in weights.items():
        arr = np.load(pipe_dir / f"oof_{model_name}.npy")
        g_oof += w * ecdf_transform(arr)(arr)
    p_oof = ecdf_transform(g_oof)(g_oof)

    b_oof = 0.7 * o_oof + 0.2 * s_oof + 0.1 * p_oof
    blend_auc = roc_auc_score(y, b_oof)
    assert abs(blend_auc - t1_facts["clean_blend_oof_auc"]) < 1e-4, f"Unexpected blend AUC: {blend_auc}"

    # Evaluate across fraction predicted positive (0.25 to 0.85), including exact reference rates
    rates = np.unique(np.r_[np.linspace(0.25, 0.85, 241), 0.407, 0.4125, 0.5933, 0.624])
    f2_bin = np.array([fbeta_score(y, b_oof >= np.quantile(b_oof, 1 - r), beta=2) for r in rates])
    f2_mac = np.array(
        [fbeta_score(y, b_oof >= np.quantile(b_oof, 1 - r), beta=2, average="macro") for r in rates]
    )

    idx_mac = int(np.argmax(f2_mac))
    r_mac = float(rates[idx_mac])
    v_mac = float(f2_mac[idx_mac])

    r_bin = t1_facts["clean_blend_oof_pos_rate_opt"]
    v_bin = t1_facts["clean_blend_oof_binf2_opt"]
    thr_bin = t1_facts["clean_blend_fixed_thr"]
    r_test = t1_facts["test_pos_rate"]
    n_test_pos = t1_facts["test_pos_count"]

    fig, ax = plt.subplots(figsize=(9.2, 4.35))

    ax.plot(
        rates * 100,
        f2_bin,
        color=NAVY,
        lw=2.3,
        label="OOF Binary F2 (positive class, LB metric)",
    )
    ax.plot(
        rates * 100,
        f2_mac,
        color=TEAL,
        lw=2.0,
        ls="--",
        label="OOF Macro F2 (both classes unweighted)",
    )

    # 1. Macro F2 optimum (~41.3% in final_v2_avg3 / 40.7% on clean ECDF blend)
    ax.scatter([r_mac * 100], [v_mac], color=TEAL, s=55, zorder=6, edgecolor="white", lw=1.0)
    ax.axvline(r_mac * 100, color=TEAL, ls=":", lw=1.0, alpha=0.8, zorder=2)
    ax.annotate(
        f"Macro F2 optimum (~41.3% v2 / {r_mac*100:.1f}% blend)\n"
        f"OOF Macro F2 = {v_mac:.4f}",
        xy=(r_mac * 100, v_mac),
        xytext=(27.0, 0.818),
        ha="left",
        va="center",
        fontsize=7.9,
        color="#0F766E",
        fontweight="bold",
        zorder=10,
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=TEAL, alpha=1.0, lw=0.8),
    )

    # 2. OOF Binary F2 optimum (62.4% at ECDF thr 0.3769, Binary F2 = 0.84007)
    ax.scatter([r_bin * 100], [v_bin], color=NAVY, s=62, zorder=7, edgecolor="white", lw=1.0)
    ax.axvline(r_bin * 100, color=NAVY, ls=":", lw=1.0, alpha=0.7, zorder=2)
    ax.annotate(
        f"OOF Binary F2 optimum: {r_bin*100:.1f}%\n"
        f"Binary F2 = {v_bin:.5f} (ECDF thr = {thr_bin:.4f})",
        xy=(r_bin * 100, v_bin),
        xytext=(64.8, 0.856),
        ha="left",
        va="center",
        fontsize=8.0,
        color=NAVY,
        fontweight="bold",
        zorder=10,
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor=NAVY, alpha=1.0, lw=0.8),
    )

    # 3. Final clean test positive rate (59.33% = 15,359 rows)
    ax.axvline(r_test * 100, color=CORAL, ls="-.", lw=1.4, alpha=0.75, zorder=2)
    ax.annotate(
        f"Final clean test pos. rate: {r_test*100:.2f}%\n"
        f"({n_test_pos:,} / {facts['meta']['submission_rows']:,} rows at thr {thr_bin:.4f};\n"
        f"lower test missingness: {t1_facts['mean_missing_per_row_test']} vs {t1_facts['mean_missing_per_row_train_labeled']})",
        xy=(r_test * 100, 0.768),
        xytext=(65.8, 0.742),
        ha="left",
        va="center",
        fontsize=7.8,
        color=CORAL,
        fontweight="bold",
        zorder=10,
        arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF7ED", edgecolor=CORAL, alpha=1.0, lw=0.8),
    )

    # 4. Isolated LB probe callout (sub_v10d -> probe_v10e: +0.0128 LB)
    probe = t1_facts["lb_probe_macro_to_binf2"]
    ax.annotate(
        "",
        xy=(63.0, 0.678),
        xytext=(41.3, 0.678),
        zorder=8,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.7),
    )
    ax.text(
        52.15,
        0.654,
        f"Isolated LB probe (sub_v10d -> probe_v10e):\n"
        f"+{probe['delta_lb']:.4f} LB ({probe['sub_v10d_macro_41pct']:.5f} -> {probe['probe_v10e_binf2_63pct']:.5f})",
        ha="center",
        va="top",
        fontsize=7.9,
        color=BLUE,
        fontweight="bold",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#F8FAFC", edgecolor=BLUE, alpha=1.0, lw=0.9),
    )

    ax.set_xlim(25, 85)
    ax.set_ylim(0.58, 0.885)
    ax.set_xlabel("Fraction predicted positive (%)")
    ax.set_ylabel("OOF F2 score (clean ECDF blend, AUC = 0.89499)")
    ax.set_title("Clean ECDF Blend Operating Point: Binary F2 vs. Macro F2 Trade-off", pad=10)
    ax.legend(loc="lower left", fontsize=8.2)

    fig.tight_layout()
    out_path = OUT_DIR / "fig_t1_f2_operating_point.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    apply_style()
    facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    tr, lab, lab_mask, y, raw_cols = load_labeled_train()
    make_fig1_signal_and_zones(lab, y, raw_cols, facts)
    make_fig2_lps_identity(lab, y, facts)
    make_fig3_f2_operating_point(tr, lab_mask, y, facts)
    print("All Task 1 figures generated successfully.")


if __name__ == "__main__":
    main()
