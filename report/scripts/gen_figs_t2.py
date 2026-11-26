"""Generate publication-grade figures for Task 2 (Player Segment Classification, v2 data).

Outputs (inside report/figures/t2/):
1. fig_t2_independence_heatmap.png
2. fig_t2_latent_2x2.png
3. fig_t2_progression_and_overfit.png
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import f1_score, log_loss

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "report", "scripts"))
from plot_style import (
    NAVY, SLATE, TEAL, CORAL, AMBER, BLUE, PURPLE, GRAY, LIGHT_GRAY, RED, apply_style
)

OUT_DIR = os.path.join(ROOT, "report", "figures", "t2")
DATA_DIR = os.path.join(ROOT, "cpe342-kaggle-main", "02_multiclass_classification")

# Exact noise feature list verified against t2/research/common.py
NOISE_COLS = [
    "random_metric_1", "random_metric_2", "random_metric_3", "account_age_days",
    "responds_to_discounts", "item_type_preference_social", "avg_match_length",
    "preferred_game_mode", "region", "platform", "device_type", "payment_method",
    "language", "account_status", "player_type_tag", "skill_tier",
]

SHORT_LABELS = {
    "play_frequency": "play_freq",
    "avg_session_duration": "avg_sess_dur",
    "total_playtime_hours": "playtime_hrs",
    "login_streak": "login_streak",
    "days_since_last_login": "days_since_login",
    "total_spending_thb": "spending_thb",
    "avg_monthly_spending": "monthly_spend",
    "spending_frequency": "spend_freq",
    "friend_count": "friend_count",
    "team_play_percentage": "team_play_pct",
    "chat_activity_score": "chat_score",
    "friend_invites_sent": "invites_sent",
    "gifts_sent_received": "gifts_sent_recv",
    "ranked_participation_rate": "ranked_rate",
    "tournament_entries": "tournaments",
    "win_rate_ranked": "win_rate_ranked",
    "achievement_completion_rate": "achiev_rate",
    "collection_progress": "collection_prog",
    "rare_items_count": "rare_items",
    "speed_of_progression": "prog_speed",
    "item_type_preference_cosmetic": "pref_cosmetic",
    "item_type_preference_performance": "pref_perf",
    "peak_concurrent_hours": "peak_hours",
}

CLASS_NAMES = {
    0: "0: Casual",
    1: "1: Grinder",
    2: "2: Social",
    3: "3: Whale",
}


def load_v2_train():
    tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    y = tr["segment"].astype(int).values
    feats = [c for c in tr.columns if c not in ("player_id", "segment") and c not in NOISE_COLS]
    disc = [c for c in feats if (not pd.api.types.is_numeric_dtype(tr[c])) or tr[c].nunique() <= 12]
    cont = [c for c in feats if c not in disc]
    return tr, y, cont, disc


def make_fig1_independence_heatmap(tr, y, cont):
    """2-panel Spearman correlation heatmap: pooled vs class-weighted within-class."""
    X = tr[cont]
    n_f = len(cont)
    off_mask = ~np.eye(n_f, dtype=bool)

    r_pool = X.corr(method="spearman").values

    # Sample-weighted average across the 4 classes (matching E_Y[rho(X_i, X_j | Y)])
    counts = np.bincount(y, minlength=4)
    weights = counts / counts.sum()
    r_classes = [X.iloc[y == k].corr(method="spearman").values for k in range(4)]
    r_within = sum(w * rk for w, rk in zip(weights, r_classes))

    pool_mean = float(np.abs(r_pool[off_mask]).mean())
    pool_max = float(np.abs(r_pool[off_mask]).max())
    within_mean = float(np.abs(r_within[off_mask]).mean())
    within_max = float(np.abs(r_within[off_mask]).max())

    labels = [SHORT_LABELS.get(c, c) for c in cont]

    fig = plt.figure(figsize=(10.6, 4.6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 0.045], wspace=0.14)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    im1 = ax1.imshow(r_pool, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="equal", interpolation="nearest")
    im2 = ax2.imshow(r_within, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="equal", interpolation="nearest")

    for ax in (ax1, ax2):
        ax.grid(False)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.set_xticks(np.arange(n_f))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6.4)
        ax.set_yticks(np.arange(n_f))
        ax.tick_params(axis="both", length=2.5, pad=2)
        # Subtle domain block guides (Engagement | Spending | Social | Competitive/Prog)
        for boundary in (4.5, 7.5, 12.5):
            ax.axhline(boundary, color="#64748B", linewidth=0.55, linestyle="--", alpha=0.45)
            ax.axvline(boundary, color="#64748B", linewidth=0.55, linestyle="--", alpha=0.45)

    ax1.set_yticklabels(labels, fontsize=6.4)
    ax2.set_yticklabels([])
    ax2.tick_params(axis="y", length=0)

    ax1.set_title(
        f"Pooled Spearman (All Rows, N = {len(tr):,})\n"
        f"Off-diag mean |r| = {pool_mean:.4f}, max |r| = {pool_max:.4f}",
        fontsize=9.0, pad=8, color=NAVY
    )
    ax2.set_title(
        f"Within-Class Weighted Mean (K = 4)\n"
        f"Off-diag mean |r| = {within_mean:.4f}, max |r| = {within_max:.4f} (~{pool_mean / within_mean:.1f}x drop)",
        fontsize=9.0, pad=8, color=NAVY
    )

    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("Spearman Correlation (r)", fontsize=8.5, color="#1E293B", labelpad=6)
    cbar.ax.tick_params(labelsize=8.0)

    out_path = os.path.join(OUT_DIR, "fig_t2_independence_heatmap.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"Saved {out_path} | pool_mean={pool_mean:.5f}, pool_max={pool_max:.5f}, within_mean={within_mean:.5f}, within_max={within_max:.5f}")


def make_fig2_latent_2x2(tr, y, facts):
    """2-panel figure: 2x2 latent class medians + EM label-noise calibration & eps recovery."""
    t2 = facts["task2"]
    v1 = t2["v1_reference"]
    n_total = len(tr)

    fig = plt.figure(figsize=(10.6, 4.2))
    outer = gridspec.GridSpec(1, 2, width_ratios=[1.12, 1.0], wspace=0.32)

    # --- Left Panel: 2x2 Latent Engagement x Spending medians ---
    ax_l = fig.add_subplot(outer[0, 0])

    # Midpoints separating the 2x2 quadrants
    x_mid = 1720.0
    y_mid = 33400.0

    # Subtle quadrant shading
    ax_l.axvspan(1150, x_mid, ymin=0, ymax=1, color="#F8FAFC", zorder=0)
    ax_l.axvspan(x_mid, 2280, ymin=0, ymax=1, color="#F1F5F9", alpha=0.55, zorder=0)
    ax_l.axvline(x_mid, color=GRAY, linestyle="--", linewidth=1.1, zorder=1)
    ax_l.axhline(y_mid, color=GRAY, linestyle="--", linewidth=1.1, zorder=1)

    class_colors = {0: BLUE, 1: TEAL, 2: AMBER, 3: CORAL}
    medians = {}
    shares = {}
    for k in range(4):
        sub = tr.loc[y == k]
        medians[k] = (float(sub["total_playtime_hours"].median()), float(sub["total_spending_thb"].median()))
        shares[k] = 100.0 * len(sub) / n_total

    # Dashed 2x2 lattice connecting adjacent classes
    for a, b in [(0, 1), (2, 3), (0, 2), (1, 3)]:
        xa, ya = medians[a]
        xb, yb = medians[b]
        ax_l.plot([xa, xb], [ya, yb], color="#94A3B8", linestyle=":", linewidth=1.3, zorder=2)

    # Offsets for text callouts so there is zero overlap with dots, axes, each other, or corner labels
    label_offsets = {
        0: (10, 14, "left", "bottom"),
        1: (-8, -16, "center", "top"),
        2: (-14, 15, "center", "bottom"),
        3: (-4, -18, "center", "top"),
    }

    for k in range(4):
        xk, yk = medians[k]
        ax_l.scatter(
            [xk], [yk], s=190, color=class_colors[k], edgecolors="white",
            linewidth=1.8, zorder=5
        )
        dx, dy, ha, va = label_offsets[k]
        ax_l.annotate(
            f"{CLASS_NAMES[k]} ({shares[k]:.1f}%)\n({xk:,.0f} h, {yk / 1000:.1f}k THB)",
            xy=(xk, yk), xytext=(dx, dy), textcoords="offset points",
            ha=ha, va=va, fontsize=7.8, fontweight="bold", color=class_colors[k],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=class_colors[k], alpha=0.95, linewidth=0.9),
            zorder=6,
        )

    # Quadrant corner descriptors in muted slate placed well outside callout zones
    quad_style = dict(fontsize=7.5, color="#64748B", style="italic")
    ax_l.text(1180, 52300, "Low Engagement / High Spend", ha="left", va="top", **quad_style)
    ax_l.text(2300, 52300, "High Engagement / High Spend", ha="right", va="top", **quad_style)
    ax_l.text(1180, 17500, "Low Engagement / Low Spend", ha="left", va="bottom", **quad_style)
    ax_l.text(2300, 17500, "High Engagement / Low Spend", ha="right", va="bottom", **quad_style)

    ax_l.set_xlim(1150, 2330)
    ax_l.set_ylim(16800, 53200)
    ax_l.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_l.yaxis.set_major_formatter(FuncFormatter(lambda yv, _: f"{int(yv // 1000)}k"))
    ax_l.set_xlabel("Median Total Playtime (hours)", fontweight="bold")
    ax_l.set_ylabel("Median Total Spending (THB)", fontweight="bold")
    ax_l.set_title("2x2 Latent Structure of Class Medians (v2 Train)", pad=10)

    # --- Right Panel: Calibration Log-Loss (top) + EM Flip Rate Recovery (bottom) ---
    right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], height_ratios=[1.95, 1.05], hspace=0.58)
    ax_ll = fig.add_subplot(right_gs[0, 0])
    ax_eps = fig.add_subplot(right_gs[1, 0])

    # Top: Log-Loss horizontal lollipop / dot-segment from baseline reference
    ll_models = ["EM-NB (v2)", "EM-NB (v1)", "Standard NB (v1)"]
    ll_vals = [float(t2["v2_emnb_oof_logloss"]), float(v1["emnb_llsel_v1_logloss"]), 0.6243]
    ll_colors = [TEAL, BLUE, CORAL]
    y_pos = np.arange(len(ll_models))

    # Draw stems from left axis edge 0.585 to avoid bar-area distortion on zoomed axis
    x_left_ll = 0.585
    ax_ll.axvline(0.6243, color=CORAL, linestyle=":", linewidth=1.1, alpha=0.65, zorder=1)
    for yi, val, col in zip(y_pos, ll_vals, ll_colors):
        ax_ll.hlines(yi, x_left_ll, val, color=col, linewidth=3.2, alpha=0.75, zorder=2)
        ax_ll.scatter([val], [yi], s=95, color=col, edgecolors="white", linewidth=1.3, zorder=4)

    # Text annotations with delta vs Standard NB v1
    ax_ll.text(ll_vals[2] + 0.0014, 2, f"{ll_vals[2]:.4f}", va="center", ha="left", fontsize=8.5, fontweight="bold", color=CORAL)
    ax_ll.text(ll_vals[1] + 0.0014, 1, f"{ll_vals[1]:.4f} (-0.0250)", va="center", ha="left", fontsize=8.3, fontweight="bold", color=BLUE)
    ax_ll.text(ll_vals[0] + 0.0014, 0, f"{ll_vals[0]:.4f} (-0.0268)", va="center", ha="left", fontsize=8.3, fontweight="bold", color=TEAL)

    # Drop arrow illustrating v1 calibration gain
    ax_ll.annotate(
        "", xy=(ll_vals[1], 1.38), xytext=(ll_vals[2], 1.38),
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.3)
    )
    ax_ll.text(
        (ll_vals[1] + ll_vals[2]) / 2, 1.46, "-4.0% Log-Loss",
        ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=NAVY
    )

    ax_ll.set_yticks(y_pos)
    ax_ll.set_yticklabels(ll_models, fontsize=8.5, fontweight="bold")
    ax_ll.set_xlim(x_left_ll, 0.638)
    ax_ll.set_ylim(-0.45, 2.45)
    ax_ll.set_xlabel("OOF Multi-Class Log-Loss (lower is better)", fontsize=8.8)
    ax_ll.set_title("EM Label-Noise Calibration Effect", pad=8)

    # Bottom: Estimated flip rate vs true 3.0% symmetric noise (zero-based bars)
    eps_hat = float(v1["estimated_label_noise_eps"])
    eps_true = float(v1["true_v1_label_noise"])
    eps_labels = ["EM Estimated\n($\\hat{\\varepsilon}$)", "True Symmetric\nNoise (v1)"]
    eps_vals = [eps_hat, eps_true]
    eps_cols = [TEAL, NAVY]
    yp_e = np.arange(2)

    ax_eps.barh(yp_e, eps_vals, height=0.52, color=eps_cols, alpha=0.88, edgecolor="white", linewidth=1.0, zorder=3)
    ax_eps.axvline(eps_true, color=NAVY, linestyle="--", linewidth=1.0, alpha=0.65, zorder=2)

    for yi, val, col in zip(yp_e, eps_vals, eps_cols):
        ax_eps.text(
            val + 0.0008, yi, f"{val:.4f} ({val * 100:.2f}%)",
            va="center", ha="left", fontsize=8.2, fontweight="bold", color=col
        )

    ax_eps.set_yticks(yp_e)
    ax_eps.set_yticklabels(eps_labels, fontsize=8.2, fontweight="bold")
    ax_eps.set_xlim(0.0, 0.043)
    ax_eps.set_ylim(-0.5, 1.55)
    ax_eps.set_xlabel("Symmetric Label-Flip Rate (fraction of rows)", fontsize=8.8)
    ax_eps.set_title("Unconditional Flip Rate Recovery (95.0% of True 3%)", fontsize=9.8, pad=6)

    out_path = os.path.join(OUT_DIR, "fig_t2_latent_2x2.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"Saved {out_path} | medians={medians}")


def compute_v2_oof_metrics(y):
    """Compute exact v2 OOF Macro F1 metrics directly from saved OOF probability arrays."""
    oof_nb = np.load(os.path.join(ROOT, "t2", "preds", "v2data", "nb_oof.npy"))
    oof_lr = np.load(os.path.join(ROOT, "work", "t2v2", "blind", "out", "logreg_spline_A_oof.npy"))
    oof_mlp = np.load(os.path.join(ROOT, "work", "t2v2", "blind", "out", "mlp_torch_A_oof.npy"))

    blend_oof = 0.87 * oof_nb + 0.08 * oof_lr + 0.05 * oof_mlp

    # Fixed-point affine F1 rule (identical to work/t2v2/make_t2v2.py)
    n_rows = len(y)
    pi = np.bincount(y, minlength=4) / n_rows
    q = pi.copy()
    fk = np.full(4, 0.75)
    for _ in range(8):
        aa = 2.0 / (pi + q)
        bb = -fk / (pi + q)
        pred_aff = (aa * blend_oof + bb).argmax(1)
        q = np.bincount(pred_aff, minlength=4) / n_rows
        fk = np.array([2.0 * ((pred_aff == k) & (y == k)).mean() / (pi[k] + q[k]) for k in range(4)])

    f1_mlp = float(f1_score(y, oof_mlp.argmax(1), average="macro"))
    f1_lr = float(f1_score(y, oof_lr.argmax(1), average="macro"))
    f1_nb = float(f1_score(y, oof_nb.argmax(1), average="macro"))
    f1_blend = float(f1_score(y, blend_oof.argmax(1), average="macro"))
    f1_affine = float(f1_score(y, pred_aff, average="macro"))
    rows_changed = int((pred_aff != blend_oof.argmax(1)).sum())
    ll_nb = float(log_loss(y, oof_nb))

    return {
        "mlp": f1_mlp,
        "spline_lr": f1_lr,
        "em_nb": f1_nb,
        "blend_argmax": f1_blend,
        "blend_affine": f1_affine,
        "rows_changed": rows_changed,
        "em_nb_logloss": ll_nb,
    }


def make_fig3_progression_and_overfit(y, facts):
    """2-panel figure: Train vs CV overfitting diagnostic (v1) + v2 Clean OOF progression."""
    v1 = facts["task2"]["v1_reference"]
    v2_m = compute_v2_oof_metrics(y)

    fig = plt.figure(figsize=(11.2, 4.3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.02, 1.08], wspace=0.50)

    # --- Left Panel: Train In-Sample vs 5-Fold CV Macro F1 (v1 ablation) ---
    ax1 = fig.add_subplot(gs[0, 0])

    # Order bottom-to-top so Final EM-NB sits at top
    models_l = ["LightGBM\n(Raw)", "Baseline\nOptuna NB", "Final EM-NB\n(LL-sel, v1)"]
    cv_nested = [float(v1["lgbm_raw_f1m"]), float(v1["nb_optuna_nested_f1m"]), float(v1["emnb_llsel_v1_f1m"])]
    train_vals = [float(v1["lgbm_train_insample_f1m"]), 0.7578, 0.7578]  # Train F1 from t2/research/REPORT.md sec 8
    opt_cv_nb = float(v1["nb_optuna_optimistic_f1m"])

    yp = np.arange(len(models_l))

    # Connect CV to Train with gap segments
    gap_colors = [RED, SLATE, TEAL]
    for i in range(len(models_l)):
        ax1.hlines(yp[i], cv_nested[i], train_vals[i], color=gap_colors[i], linewidth=3.4, alpha=0.55, zorder=2)

    # Markers
    ax1.scatter(
        cv_nested, yp, s=90, color=TEAL, edgecolors="white", linewidth=1.3,
        marker="o", label="Honest 5-Fold CV (Nested)", zorder=4
    )
    ax1.scatter(
        [opt_cv_nb], [1], s=80, color=AMBER, edgecolors="white", linewidth=1.1,
        marker="D", label="Optimistic CV (Non-Nested)", zorder=5
    )
    ax1.scatter(
        train_vals, yp, s=82, color=NAVY, edgecolors="white", linewidth=1.2,
        marker="s", label="Train In-Sample", zorder=4
    )

    # Direct labels (positioned cleanly above/below each row)
    # Row 0: LightGBM
    gap_lgb = train_vals[0] - cv_nested[0]
    ax1.text(cv_nested[0], 0 - 0.16, f"{cv_nested[0]:.4f}", ha="center", va="top", fontsize=8.0, fontweight="bold", color=TEAL)
    ax1.text(train_vals[0], 0 - 0.16, f"{train_vals[0]:.4f}", ha="center", va="top", fontsize=8.0, fontweight="bold", color=NAVY)
    ax1.text(
        (cv_nested[0] + train_vals[0]) / 2, 0 + 0.13, f"Overfit gap: +{gap_lgb:.4f}",
        ha="center", va="bottom", fontsize=8.1, fontweight="bold", color=RED,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FEF2F2", edgecolor=RED, alpha=0.92, linewidth=0.8)
    )

    # Row 1: Baseline Optuna NB
    ax1.text(cv_nested[1] - 0.0012, 1 - 0.16, f"{cv_nested[1]:.4f}", ha="right", va="top", fontsize=7.8, fontweight="bold", color=TEAL)
    ax1.text(opt_cv_nb, 1 + 0.16, f"Opt: {opt_cv_nb:.4f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=AMBER)
    ax1.text(train_vals[1] + 0.0014, 1 - 0.16, f"{train_vals[1]:.4f}", ha="left", va="top", fontsize=7.8, fontweight="bold", color=NAVY)

    # Row 2: Final EM-NB v1
    gap_nb = train_vals[2] - cv_nested[2]
    ax1.text(cv_nested[2] - 0.0012, 2 - 0.16, f"{cv_nested[2]:.4f}", ha="right", va="top", fontsize=8.0, fontweight="bold", color=TEAL)
    ax1.text(train_vals[2] + 0.0014, 2 - 0.16, f"{train_vals[2]:.4f}", ha="left", va="top", fontsize=8.0, fontweight="bold", color=NAVY)
    ax1.text(
        (cv_nested[2] + train_vals[2]) / 2, 2 + 0.16, f"Gap: +{gap_nb:.4f}",
        ha="center", va="bottom", fontsize=7.9, fontweight="bold", color=TEAL
    )

    ax1.set_yticks(yp)
    ax1.set_yticklabels(models_l, fontsize=8.6, fontweight="bold")
    ax1.set_xlim(0.745, 0.799)
    ax1.set_ylim(-0.52, 2.55)
    ax1.set_xlabel("Macro F1 Score", fontweight="bold")
    ax1.set_title("Train vs 5-Fold CV Generalization Gap (v1)", pad=10)
    ax1.legend(loc="upper right", fontsize=7.6, frameon=True, borderpad=0.4)

    # --- Right Panel: v2 Clean Model Progression (from OOF arrays) ---
    ax2 = fig.add_subplot(gs[0, 1])

    stage_names = [
        "PyTorch MLP",
        "Spline Logistic Reg.",
        "EM-Naive Bayes (LL-sel)",
        "Blend (87/8/5) Argmax",
        "Blend + Affine F1 Rule",
    ]
    stage_vals = [
        round(v2_m["mlp"], 5),
        round(v2_m["spline_lr"], 5),
        round(v2_m["em_nb"], 5),
        round(v2_m["blend_argmax"], 5),
        round(v2_m["blend_affine"], 5),
    ]
    stage_cols = [GRAY, SLATE, BLUE, NAVY, TEAL]
    y2 = np.arange(len(stage_names))

    x_anchor = 0.750
    ax2.axvline(stage_vals[2], color=BLUE, linestyle=":", linewidth=1.0, alpha=0.55, zorder=1)

    for yi, val, col in zip(y2, stage_vals, stage_cols):
        ax2.hlines(yi, x_anchor, val, color=col, linewidth=3.8, alpha=0.82, zorder=2)
        sz = 125 if yi == 4 else 90
        ax2.scatter([val], [yi], s=sz, color=col, edgecolors="white", linewidth=1.4, zorder=4)

        if yi == 0:
            txt = f"{val:.5f}"
        else:
            delta = val - stage_vals[yi - 1]
            txt = f"{val:.5f} (+{delta:.5f})"
        ax2.text(
            val + 0.00026, yi, txt,
            va="center", ha="left", fontsize=8.0,
            fontweight="bold" if yi >= 2 else "normal",
            color=col if yi >= 2 else "#334155",
        )

    ax2.set_yticks(y2)
    ax2.set_yticklabels(stage_names, fontsize=8.4, fontweight="bold")
    ax2.set_xlim(x_anchor, 0.7634)
    ax2.set_ylim(-0.82, 4.55)
    ax2.set_xlabel("v2 Clean OOF Macro F1 Score (no player_id)", fontweight="bold")
    ax2.set_title("v2 Clean Feature-Only Model Progression", pad=10)

    # Compact footer badge inside bottom-right whitespace
    ax2.text(
        0.98, 0.04,
        f"Affine rule shifts {v2_m['rows_changed']:,} / {len(y):,} OOF rows (1.98%)",
        transform=ax2.transAxes, ha="right", va="bottom", fontsize=7.6, color=NAVY,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0FDFA", edgecolor=TEAL, alpha=0.92, linewidth=0.8),
    )

    out_path = os.path.join(OUT_DIR, "fig_t2_progression_and_overfit.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"Saved {out_path} | v2_metrics={v2_m}")


def main():
    apply_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(ROOT, "report", "facts_ledger.json"), encoding="utf-8") as f:
        facts = json.load(f)

    tr, y, cont, _ = load_v2_train()
    make_fig1_independence_heatmap(tr, y, cont)
    make_fig2_latent_2x2(tr, y, facts)
    make_fig3_progression_and_overfit(y, facts)


if __name__ == "__main__":
    main()
