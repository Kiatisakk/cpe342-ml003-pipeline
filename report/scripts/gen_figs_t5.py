"""Generate publication-grade figures for Task 5 (Account Security Monitoring, v2 data)."""
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from plot_style import (  # noqa: E402
    AMBER,
    BLUE,
    CORAL,
    GRAY,
    LIGHT_GRAY,
    NAVY,
    PURPLE,
    RED,
    SLATE,
    TEAL,
    apply_style,
)

PERIODS = [1, 2, 3, 4]


def load_t5_artifacts():
    """Load Task 5 v2 data, pipeline outputs, TMAE scores, and facts ledger."""
    x_df = pd.read_csv(os.path.join(REPO_ROOT, "cpe342-kaggle-main/05_anomaly_detection/test.csv"))
    grp_df = pd.read_csv(os.path.join(REPO_ROOT, "work/t5v2/out/t5_groups.csv"))
    lsnan_df = pd.read_csv(os.path.join(REPO_ROOT, "work/t5v2/out/lowstab_nan_scores.csv"))
    tmae_df = pd.read_csv(os.path.join(REPO_ROOT, "work/t5v2/tmae/task5_tmae_scores.csv"))
    final_df = pd.read_csv(os.path.join(REPO_ROOT, "t5/preds/v2data/t5v2_union_box_lsnan.csv"))
    with open(os.path.join(REPO_ROOT, "work/t5v2/out/t5_pipeline_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    with open(os.path.join(REPO_ROOT, "report/facts_ledger.json"), encoding="utf-8") as f:
        facts = json.load(f)["task5"]
    return x_df, grp_df, lsnan_df, tmae_df, final_df, summary, facts


def compute_signals_and_groups(x_df, grp_df, lsnan_df, final_df, summary, facts):
    """Recompute shock statistics, 4-period means, and verify exact group partitions."""
    per_bases = sorted({c[:-2] for c in x_df.columns if c[-2:] in ("_1", "_2", "_3", "_4")})
    cont_bases = [
        b for b in per_bases
        if b not in ("login_lat", "login_lon", "device_count", "refund_count", "guild_changes")
    ]
    feat_means = pd.DataFrame(
        {b: x_df[[f"{b}_{p}" for p in PERIODS]].mean(axis=1) for b in cont_bases}
    )
    feat_means["account_age_days"] = x_df["account_age_days"]
    feat_means["behavior_stability"] = x_df["behavior_stability"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lat = np.stack([x_df[f"login_lat_{p}"] for p in PERIODS], axis=1).astype(float)
        lon = np.stack([x_df[f"login_lon_{p}"] for p in PERIODS], axis=1).astype(float)
        bad_lat = (np.abs(lat) > 90).any(axis=1)
        lt = np.where(np.abs(lat) > 90, np.nan, lat)
        hl, hn = np.nanmean(lt[:, :2], axis=1), np.nanmean(lon[:, :2], axis=1)
        jump = np.nan_to_num(
            np.nanmax(
                np.hypot(
                    lt[:, 2:] - hl[:, None],
                    (lon[:, 2:] - hn[:, None] + 180) % 360 - 180,
                ),
                axis=1,
            )
        )

        z_cols = []
        for b in [b for b in per_bases if b not in ("login_lat", "login_lon")]:
            mat = x_df[[f"{b}_{p}" for p in PERIODS]].astype(float).values
            ch = np.nanmax(np.abs(mat[:, 2:] - np.nanmean(mat[:, :2], axis=1, keepdims=True)), axis=1)
            med = np.nanmedian(ch)
            mad = 1.4826 * np.nanmedian(np.abs(ch - med)) + 1e-9
            z_cols.append((ch - med) / mad)

    z_mat = np.nan_to_num(np.stack(z_cols, axis=1))
    z2 = np.sort(z_mat, axis=1)[:, -2]
    shock = (jump > 5) | bad_lat | (z2 > 6)

    # Verify against t5_groups.csv and facts_ledger.json
    assert (shock == (grp_df["rule"].values == 1)).all()
    assert int(shock.sum()) == facts["v2_detectors"]["A_shock"]

    bot = (~shock) & (feat_means["ip_hash_entropy"].values < 0.238)
    lowact = (~shock) & (feat_means["activity_level"].values < 0.345)
    lowcomm = (~shock) & (feat_means["communication_frequency"].values < 0.169)
    lowstab = (~shock) & (feat_means["behavior_stability"].values < 0.473)
    assert (bot == (grp_df["bot"].values == 1)).all()
    assert (lowact == (grp_df["lowact"].values == 1)).all()
    assert (lowcomm == (grp_df["lowcomm"].values == 1)).all()
    assert (lowstab == (grp_df["lowstab"].values == 1)).all()

    gapflag = bot | lowact | lowcomm | lowstab
    assert int(gapflag.sum()) == facts["v2_detectors"]["B_gap_total"]

    rest = (~shock) & (~gapflag)
    bounds = summary["components"][0]["box"]
    inbox = rest.copy()
    for col, (lo, hi) in bounds.items():
        mat = x_df[[f"{col}_{p}" for p in PERIODS]].values.astype(float)
        inbox &= (((mat >= lo) & (mat <= hi)) | np.isnan(mat)).all(axis=1)
    assert int(inbox.sum()) == facts["v2_detectors"]["C_acckda_box"]

    lsnan_added = (lsnan_df["post_lowstab_nan"].values > 0.1) & (~(shock | gapflag | inbox))
    assert int(lsnan_added.sum()) == facts["v2_detectors"]["D_hidden_lowstab_added"]

    final = shock | gapflag | inbox | lsnan_added
    assert (final.astype(int) == final_df["pred"].values).all()
    assert int(final.sum()) == facts["total_flagged"]

    def ext(col):
        v = x_df[col].values
        lo, hi = np.nanquantile(v[~shock], [0.0005, 0.9995])
        return (v < lo) | (v > hi)

    takeover = shock & (ext("currency_transfers_4") | ext("play_pattern_consistency_4"))
    selling = shock & (~takeover) & (
        ext("skill_trajectory_4") | ext("skill_rating_4") | ext("friend_removes_4")
    )
    vpn = shock & (~takeover) & (~selling) & (
        ext("purchase_frequency_4") | ext("purchase_amount_mean_4")
    )
    assert int(takeover.sum()) == facts["v2_detectors"]["A_subtypes"]["takeover"]["count"]
    assert int(selling.sum()) == facts["v2_detectors"]["A_subtypes"]["selling"]["count"]
    assert int(vpn.sum()) == facts["v2_detectors"]["A_subtypes"]["vpn"]["count"]

    return {
        "feat_means": feat_means,
        "z_mat": z_mat,
        "z2": z2,
        "jump": jump,
        "shock": shock,
        "takeover": takeover,
        "selling": selling,
        "vpn": vpn,
        "bot": bot,
        "lowact": lowact,
        "lowcomm": lowcomm,
        "lowstab": lowstab,
        "gapflag": gapflag,
        "rest": rest,
        "inbox": inbox,
        "bounds": bounds,
        "lsnan_added": lsnan_added,
        "final": final,
    }


def plot_fig1_shock_and_gaps(sig, facts, out_path):
    """Figure 1: Stage A shock threshold, 4 Stage B empty-gap clusters, and 4-stage Union."""
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.3))
    fig.subplots_adjust(hspace=0.42, wspace=0.32, top=0.89, bottom=0.09, left=0.07, right=0.98)

    # Panel (a): 2nd-largest robust z (clipped at 60)
    ax0 = axes[0, 0]
    z_clip = np.clip(sig["z2"], 0, 60)
    bins_z = np.linspace(0, 60, 121)  # width = 0.5, exact edge at z = 6.0
    norm_mask = z_clip <= 6.0
    shock_z_mask = z_clip > 6.0
    ax0.hist(z_clip[norm_mask], bins=bins_z, log=True, color=SLATE, alpha=0.82, label="z2 <= 6")
    ax0.hist(z_clip[shock_z_mask], bins=bins_z, log=True, color=CORAL, alpha=0.90, label="z2 > 6 (816)")
    ax0.axvline(6.0, color=RED, linestyle="--", linewidth=1.6, zorder=4)
    ax0.set_ylim(0.7, 3.5e4)
    ax0.set_xlim(0, 61)
    ax0.set_title("(a) Stage A: Temporal Shock (Periods 3-4)")
    ax0.set_xlabel("2nd-largest robust z (clipped at 60)")
    ax0.set_ylabel("Accounts (log scale)")
    ax0.annotate(
        "Cut: z > 6\n816 metric shock\n(+11 geo/lat = 827)",
        xy=(6.0, 2.2e3),
        xytext=(15.5, 5.2e3),
        fontsize=8.0,
        fontweight="bold",
        color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FEF2F2", edgecolor=RED, linewidth=0.9),
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2, connectionstyle="arc3,rad=-0.12"),
    )

    # Panels (b)-(e): 4 empty-gap clusters on non-shock rows (~shock)
    gap_specs = [
        (axes[0, 1], "activity_level", 0.345, 265, "(b) Stage B1: Low Activity Cluster", "activity_level (4-period mean)"),
        (axes[0, 2], "communication_frequency", 0.169, 263, "(c) Stage B2: Low Comm. Cluster", "communication_frequency (4-period mean)"),
        (axes[1, 0], "behavior_stability", 0.473, 246, "(d) Stage B3: Low Stability Cluster", "behavior_stability (static)"),
        (axes[1, 1], "ip_hash_entropy", 0.238, 207, "(e) Stage B4: Bot Cluster", "ip_hash_entropy (4-period mean)"),
    ]

    non_shock = ~sig["shock"]
    for idx, (ax, col, cut, expected_n, title, xlabel) in enumerate(gap_specs):
        vals = sig["feat_means"][col].values[non_shock]
        vals = vals[~np.isnan(vals)]
        low_vals = vals[vals < cut]
        high_vals = vals[vals >= cut]
        assert len(low_vals) == expected_n, f"{col}: {len(low_vals)} != {expected_n}"

        gap_lo, gap_hi = float(low_vals.max()), float(high_vals.min())
        bins = np.linspace(float(vals.min()), float(vals.max()), 81)

        ax.axvspan(gap_lo, gap_hi, color="#FEF3C7", alpha=0.75, zorder=0, label="Empty gap")
        ax.hist(high_vals, bins=bins, log=True, color=SLATE, alpha=0.80, zorder=2)
        ax.hist(low_vals, bins=bins, log=True, color=TEAL, alpha=0.92, zorder=3)
        ax.axvline(cut, color=NAVY, linestyle="--", linewidth=1.5, zorder=4)

        ax.set_ylim(0.7, 1.8e4)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if idx == 2:
            ax.set_ylabel("Accounts (log scale)")

        x_span = float(vals.max() - vals.min())
        text_x = min(cut + 0.10 * x_span, float(vals.max()) * 0.56)
        ax.annotate(
            f"Cut < {cut:.3f}\nn = {expected_n} rows\nGap [{gap_lo:.2f}, {gap_hi:.2f}]",
            xy=(cut, 950),
            xytext=(text_x, 3.6e3),
            fontsize=7.9,
            fontweight="bold",
            color=NAVY,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="#F0FDFA", edgecolor=TEAL, linewidth=0.9),
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.1, connectionstyle="arc3,rad=-0.10"),
        )

    # Panel (f): Mini summary bar of the 4-stage Union
    ax5 = axes[1, 2]
    stages = [
        "D: Hidden\nLowStab",
        "C: Acc/KDA\nBox",
        "B: 4 Gap\nClusters",
        "A: Temporal\nShock",
    ]
    counts = [
        facts["v2_detectors"]["D_hidden_lowstab_added"],
        facts["v2_detectors"]["C_acckda_box"],
        facts["v2_detectors"]["B_gap_total"],
        facts["v2_detectors"]["A_shock"],
    ]
    colors = [AMBER, PURPLE, TEAL, CORAL]
    y_pos = np.arange(len(stages))
    bars = ax5.barh(y_pos, counts, height=0.56, color=colors, edgecolor="none", alpha=0.92)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(stages, fontsize=8.0, fontweight="bold")
    ax5.set_ylim(-1.25, 3.55)
    ax5.set_xlim(0, 1420)
    ax5.set_xlabel("Flagged Accounts (0 overlaps)")
    ax5.set_title("(f) 4-Stage Union Summary")
    ax5.grid(axis="y", visible=False)

    total_flagged = facts["total_flagged"]
    for bar, cnt in zip(bars, counts):
        share = 100.0 * cnt / total_flagged
        ax5.text(
            cnt + 24,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt:,} ({share:.1f}%)",
            va="center",
            ha="left",
            fontsize=8.1,
            fontweight="bold",
            color="#1E293B",
        )

    ax5.text(
        0.50,
        0.05,
        f"Union Total: {total_flagged:,} / {facts['test_rows']:,} ({100*facts['flagged_rate']:.2f}%)\n"
        f"Clean residual max |r|: {facts['independence_clean_max_r']:.3f}",
        transform=ax5.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
        color=NAVY,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F1F5F9", edgecolor=NAVY, linewidth=0.9),
    )

    fig.suptitle(
        "Task 5 Unsupervised Discovery: Shock Cut, Empty-Gap Clusters & 4-Stage Union (v2 Test, N = 25,889)",
        fontsize=11.5,
        fontweight="bold",
        color=NAVY,
        y=0.965,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_fig2_acckda_and_subtypes(sig, facts, out_path):
    """Figure 2: Acc/KDA 3D box scatter + independent security-flag validation of shock subtypes."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    fig.subplots_adjust(wspace=0.26, top=0.85, bottom=0.14, left=0.07, right=0.98)

    # Left panel: 2D scatter on rest rows (4-period mean accuracy_std vs kda_std)
    rest_bg = sig["rest"] & (~sig["inbox"])
    inbox = sig["inbox"]
    f_means = sig["feat_means"]

    bg_x = f_means.loc[rest_bg, "accuracy_std"].values
    bg_y = f_means.loc[rest_bg, "kda_std"].values
    valid_bg = (~np.isnan(bg_x)) & (~np.isnan(bg_y))

    ax_l.scatter(
        bg_x[valid_bg],
        bg_y[valid_bg],
        s=5,
        color=GRAY,
        alpha=0.32,
        edgecolors="none",
        zorder=1,
        label=f"Remaining normal (n = {int(rest_bg.sum()):,})",
    )
    ax_l.scatter(
        f_means.loc[inbox, "accuracy_std"].values,
        f_means.loc[inbox, "kda_std"].values,
        s=15,
        color=RED,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.30,
        zorder=3,
        label=f"3D Box cluster: acckda (n = {int(inbox.sum())})",
    )

    acc_lo, acc_hi = sig["bounds"]["accuracy_std"]
    kda_lo, kda_hi = sig["bounds"]["kda_std"]
    rect = mpatches.Rectangle(
        (acc_lo, kda_lo),
        acc_hi - acc_lo,
        kda_hi - kda_lo,
        fill=False,
        edgecolor=NAVY,
        linestyle="--",
        linewidth=1.7,
        zorder=4,
        label="Box [8.96, 12.03] x [0.70, 1.00]",
    )
    ax_l.add_patch(rect)

    ax_l.set_title("(a) Stage C: Joint Tail Box Cluster (z = 20.9, 99.7% precision)")
    ax_l.set_xlabel("accuracy_std (4-period mean)")
    ax_l.set_ylabel("kda_std (4-period mean)")
    ax_l.set_xlim(1.8, 13.0)
    ax_l.set_ylim(0.10, 1.07)
    ax_l.legend(loc="upper left", fontsize=7.9, markerscale=1.3)

    ax_l.annotate(
        "244 accounts inside 3D per-period box\n"
        "(+ performance_stability in [0.85, 0.95])\n"
        "Null-calibrated precision: 99.7%",
        xy=(10.5, 0.70),
        xytext=(2.3, 0.17),
        fontsize=7.9,
        fontweight="bold",
        color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FEF2F2", edgecolor=RED, linewidth=0.9),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2, connectionstyle="arc3,rad=0.18"),
    )

    # Right panel: Grouped bar chart of unused security flags across shock subtypes
    subtypes_meta = [
        ("takeover", "Takeover\n(n = 310)"),
        ("selling", "Selling\n(n = 311)"),
        ("vpn", "VPN\n(n = 198)"),
        ("non_shock", "Non-Shock\n(n = 25,062)"),
    ]
    flags_meta = [
        ("password_changed", "password_changed", NAVY),
        ("mass_item_sale", "mass_item_sale", TEAL),
        ("vpn_usage", "vpn_usage", AMBER),
    ]

    sub_data = facts["v2_detectors"]["A_subtypes"]
    x_idx = np.arange(len(subtypes_meta))
    width = 0.27

    for i, (f_key, f_label, color) in enumerate(flags_meta):
        rates = [100.0 * sub_data[s_key][f_key] for s_key, _ in subtypes_meta]
        offset = (i - 1) * width
        bars = ax_r.bar(x_idx + offset, rates, width=width, color=color, alpha=0.92, label=f_label)
        for j, (bar, val) in enumerate(zip(bars, rates)):
            x_txt = bar.get_x() + bar.get_width() / 2
            # Small horizontal nudges to keep adjacent labels strictly non-overlapping
            if j == 0 and i == 1:
                x_txt += 0.04
            elif j == 2 and i == 0:
                x_txt -= 0.03
            elif j == 2 and i == 1:
                x_txt += 0.02
            ax_r.text(
                x_txt,
                val + 1.8,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.0,
                fontweight="bold",
                color="#1E293B",
            )

    ax_r.set_xticks(x_idx)
    ax_r.set_xticklabels([lbl for _, lbl in subtypes_meta], fontsize=8.5, fontweight="bold")
    ax_r.set_ylim(0, 118)
    ax_r.set_ylabel("Unused Security Flag Rate (%)")
    ax_r.set_title("(b) Shock Subtypes vs. Unused Security Flags")
    ax_r.grid(axis="x", visible=False)
    ax_r.legend(loc="upper center", ncol=1, fontsize=8.0, bbox_to_anchor=(0.45, 0.98))

    fig.suptitle(
        "Stage C Joint Box Discovery & Independent Validation of Stage A Shock Subtypes",
        fontsize=11.3,
        fontweight="bold",
        color=NAVY,
        y=0.96,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_fig3_baseline_capture(sig, tmae_df, out_path):
    """Figure 3: Baseline capture rate comparison at top K = 2,104 across all 9 anomaly groups."""
    f_means = sig["feat_means"]
    feat = pd.concat(
        [
            ((f_means - f_means.median()) / (f_means.quantile(0.75) - f_means.quantile(0.25))).fillna(0),
            pd.DataFrame(np.clip(sig["z_mat"], -50, 50) / 10.0),
            pd.Series(np.clip(sig["jump"], 0, 50) / 10.0, name="jump"),
        ],
        axis=1,
    )
    feat.columns = feat.columns.astype(str)
    iso_scores = -IsolationForest(
        n_estimators=500, max_samples=1024, random_state=0, n_jobs=-1
    ).fit(feat).score_samples(feat)

    tmae_scores = tmae_df["tmae"].values
    final = sig["final"]
    k_top = int(final.sum())

    def topk_mask(scores):
        return scores >= np.sort(scores)[-k_top]

    iso_top = topk_mask(iso_scores)
    tmae_top = topk_mask(tmae_scores)
    assert int(iso_top.sum()) == k_top
    assert int(tmae_top.sum()) == k_top

    iso_outside = int((iso_top & (~final)).sum())
    tmae_outside = int((tmae_top & (~final)).sum())

    groups = [
        ("Takeover", sig["takeover"]),
        ("Selling", sig["selling"]),
        ("VPN", sig["vpn"]),
        ("Bot", sig["bot"]),
        ("Low Activity", sig["lowact"]),
        ("Low Comm.", sig["lowcomm"]),
        ("Low Stability", sig["lowstab"]),
        ("Acc/KDA Box", sig["inbox"]),
        ("Hidden LowStab", sig["lsnan_added"]),
    ]

    # Reverse order for horizontal bars so Takeover is at the top and Hidden LowStab at the bottom
    groups_plot = list(reversed(groups))
    labels = [f"{name} (n={int(mask.sum())})" for name, mask in groups_plot]
    union_rates = [100.0 for _ in groups_plot]
    iso_rates = [100.0 * float(iso_top[mask].mean()) for _, mask in groups_plot]
    tmae_rates = [100.0 * float(tmae_top[mask].mean()) for _, mask in groups_plot]

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    fig.subplots_adjust(left=0.20, right=0.96, top=0.75, bottom=0.16)

    y_idx = np.arange(len(groups_plot))
    bar_h = 0.26

    bars_u = ax.barh(
        y_idx + bar_h, union_rates, height=bar_h, color=TEAL, alpha=0.94,
        label="Final 4-Stage Union (0 outside final set)"
    )
    bars_i = ax.barh(
        y_idx, iso_rates, height=bar_h, color=NAVY, alpha=0.90,
        label=f"Isolation Forest top {k_top:,} ({iso_outside:,} outside final = {100*iso_outside/k_top:.1f}%)"
    )
    bars_t = ax.barh(
        y_idx - bar_h, tmae_rates, height=bar_h, color=CORAL, alpha=0.90,
        label=f"Autoencoder (TMAE) top {k_top:,} ({tmae_outside:,} outside final = {100*tmae_outside/k_top:.1f}%)"
    )

    for bars in (bars_u, bars_i, bars_t):
        for bar in bars:
            val = bar.get_width()
            lbl = "100%" if val > 99.95 else f"{val:.1f}%"
            ax.text(
                val + 1.0,
                bar.get_y() + bar.get_height() / 2,
                lbl,
                va="center",
                ha="left",
                fontsize=7.2,
                fontweight="bold",
                color="#1E293B",
            )

    ax.set_yticks(y_idx)
    ax.set_yticklabels(labels, fontsize=8.8, fontweight="bold")
    ax.set_xlim(0, 112)
    ax.set_xlabel(f"Recall Capture Rate at Top K = {k_top:,} Flagged Accounts (%)")
    ax.grid(axis="y", visible=False)
    ax.axhline(5.5, color="#CBD5E1", linestyle=":", linewidth=1.1)
    ax.axhline(1.5, color="#CBD5E1", linestyle=":", linewidth=1.1)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.48, 1.02),
        ncol=1,
        fontsize=8.3,
        frameon=True,
    )
    fig.suptitle(
        f"Baseline Failure Modes: Recall Capture Across All 9 Anomaly Groups (Top K = {k_top:,})",
        fontsize=11.3,
        fontweight="bold",
        color=NAVY,
        y=0.965,
    )
    fig.text(
        0.58,
        0.025,
        f"Contamination in top {k_top:,}:  Isolation Forest = {iso_outside:,} non-union accounts (42.2%)   |   "
        f"Autoencoder (TMAE) = {tmae_outside:,} non-union accounts (82.8%)",
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FEF2F2", edgecolor=CORAL, linewidth=0.9),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print("Baseline Capture Summary (K = %d):" % k_top)
    print(f"  Isolation Forest outside final: {iso_outside} ({100*iso_outside/k_top:.1f}%)")
    print(f"  Autoencoder TMAE outside final: {tmae_outside} ({100*tmae_outside/k_top:.1f}%)")
    for (name, _), ir, tr in zip(groups_plot, iso_rates, tmae_rates):
        print(f"    {name:15s}: IForest={ir:5.1f}% | TMAE={tr:5.1f}%")


def main():
    apply_style()
    x_df, grp_df, lsnan_df, tmae_df, final_df, summary, facts = load_t5_artifacts()
    sig = compute_signals_and_groups(x_df, grp_df, lsnan_df, final_df, summary, facts)

    fig_dir = os.path.join(REPO_ROOT, "report/figures/t5")
    p1 = os.path.join(fig_dir, "fig_t5_shock_and_gaps.png")
    p2 = os.path.join(fig_dir, "fig_t5_acckda_box.png")
    p3 = os.path.join(fig_dir, "fig_t5_baseline_capture.png")

    plot_fig1_shock_and_gaps(sig, facts, p1)
    print("Saved:", p1)
    plot_fig2_acckda_and_subtypes(sig, facts, p2)
    print("Saved:", p2)
    plot_fig3_baseline_capture(sig, tmae_df, p3)
    print("Saved:", p3)


if __name__ == "__main__":
    main()
