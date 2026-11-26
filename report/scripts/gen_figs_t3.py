"""Generate publication-grade figures for Task 3 (Player Monthly Spending Prediction).

Outputs (inside report/figures/t3/):
  1. fig_t3_whale_and_error_budget.png
  2. fig_t3_offset_linearity.png
  3. fig_t3_model_progression_ceiling.png

Read-only inputs:
  - report/facts_ledger.json
  - cpe342-kaggle-main/03_regression/train.csv
  - t3/preds/seedbag10/repo/t3/research/oof/bag_q10p10_Q.npy
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(REPORT_DIR)
OUT_DIR = os.path.join(REPORT_DIR, "figures", "t3")

sys.path.insert(0, SCRIPT_DIR)
import plot_style as ps  # noqa: E402


def load_facts():
    ledger_path = os.path.join(REPORT_DIR, "facts_ledger.json")
    with open(ledger_path, encoding="utf-8") as f:
        return json.load(f)


def make_fig1_whale_and_error_budget(facts):
    """2-panel figure: segment concentration vs 4-part OOF error budget."""
    t3 = facts["task3"]
    seg_bd = t3["segment_breakdown_oof"]
    total_rows = float(t3["train_rows"])
    budget = t3["error_budget_from_insights"]

    seg_keys = ["0", "1", "2", "3", "-1"]
    seg_labels = ["Casual\n(Seg 0)", "Grinder\n(Seg 1)", "Social\n(Seg 2)", "Whale\n(Seg 3)", "Missing\n(NaN)"]

    row_shares = np.array([100.0 * seg_bd[k]["rows"] / total_rows for k in seg_keys])
    spend_shares = np.array([seg_bd[k]["spend_share_pct"] for k in seg_keys])
    err_shares = np.array([seg_bd[k]["err_share_pct"] for k in seg_keys])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": [1.22, 0.94], "wspace": 0.35}
    )

    # --- Panel A: Grouped bars per segment ---
    x = np.arange(len(seg_keys)) * 1.42
    w = 0.37
    b1 = ax1.bar(x - w, row_shares, width=w, color=ps.SLATE, alpha=0.82, label="Share of Rows (%)")
    b2 = ax1.bar(x, spend_shares, width=w, color=ps.TEAL, label="Share of Total Spend (%)")
    b3 = ax1.bar(x + w, err_shares, width=w, color=ps.CORAL, label="Share of OOF |Error| (%)")

    # Clean non-overlapping value labels (exact 2 decimals on Whale bars & tiny spend)
    for bar_group_idx, (bars, vals) in enumerate([(b1, row_shares), (b2, spend_shares), (b3, err_shares)]):
        for idx, (bar, v) in enumerate(zip(bars, vals)):
            if idx == 3 or v < 1.0:
                lbl = f"{v:.2f}%"
            else:
                lbl = f"{v:.1f}%"
            x_shift = -0.02 if bar_group_idx == 0 and idx in (0, 4) else (0.02 if bar_group_idx == 2 and idx in (0, 4) else 0.0)
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0 + x_shift,
                bar.get_height() + 1.2,
                lbl,
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#1E293B",
                fontweight="bold" if (idx == 3 and v > 50.0) else "normal",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(seg_labels, fontsize=8.5)
    ax1.set_ylabel("Share of Dataset Total (%)")
    ax1.set_ylim(0, 102)
    ax1.set_title("(a) Segment Concentration: Rows vs. Spend vs. Error", pad=10)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.2)
    ax1.grid(axis="x", visible=False)

    # Callout placed completely in open space above Grinder/Social bars
    whale_x = x[3]
    ax1.annotate(
        "Whales = 14.02% of rows\n87.28% of spend\n79.37% of OOF |Error|",
        xy=(whale_x - 0.1 * w, 86.0),
        xytext=(x[1] + 0.35, 56.0),
        fontsize=8.0,
        fontweight="bold",
        color=ps.NAVY,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF7ED", edgecolor=ps.CORAL, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=ps.CORAL, lw=1.25, connectionstyle="arc3,rad=-0.16"),
    )

    # --- Panel B: 4-part error budget ---
    categories = [
        "Payer multiplicative noise",
        "y = 0 predicted > 0\n(coin-flip uncertainty)",
        "Whale spikes\n(266 rows redrawn)",
        "Payers predicted 0",
    ]
    values = [
        budget["payer_mult_noise_pct"],
        budget["zero_pred_pos_pct"],
        budget["whale_spike_266rows_pct"],
        budget["payer_pred_zero_pct"],
    ]
    colors = [ps.TEAL, ps.AMBER, ps.CORAL, ps.SLATE]

    y_pos = np.arange(len(categories))
    bars = ax2.barh(y_pos, values, height=0.56, color=colors, edgecolor="none", alpha=0.92)
    ax2.invert_yaxis()

    for bar, v in zip(bars, values):
        ax2.text(
            v + 1.1,
            bar.get_y() + bar.get_height() / 2.0,
            f"{v:.1f}%",
            va="center",
            ha="left",
            fontsize=8.8,
            fontweight="bold",
            color="#1E293B",
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories, fontsize=8.5)
    ax2.set_xlabel("Share of Total OOF Absolute Error (%)")
    ax2.set_xlim(0, 59)
    ax2.set_title("(b) OOF Absolute Error Budget (4 Sources)", pad=10)
    ax2.grid(axis="y", visible=False)

    ax2.text(
        0.96,
        0.08,
        "Irreducible generator draws\n"
        "account for >97% of error;\n"
        f"10x10 bag OOF NMAE = {t3['bags']['bag_q10p10']['oof_nmae']:.5f}",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        color=ps.NAVY,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.9),
    )

    out_path = os.path.join(OUT_DIR, "fig_t3_whale_and_error_budget.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_fig2_offset_linearity():
    """2-panel figure: log(y) vs log(prev) unit-slope linearity & bounded trapezoid residual."""
    tr = pd.read_csv(os.path.join(ROOT_DIR, "cpe342-kaggle-main", "03_regression", "train.csv"))
    Q_path = os.path.join(
        ROOT_DIR, "t3", "preds", "seedbag10", "repo", "t3", "research", "oof", "bag_q10p10_Q.npy"
    )
    Q = np.load(Q_path)[: len(tr)]
    q50 = Q[:, -1]  # alpha = 0.50 conditional median given y > 0

    mask_pos = (tr["spending_30d"] > 0) & (tr["prev_month_spending"] > 0) & (tr["segment"].isin([0, 1, 2, 3]))
    pos = tr.loc[mask_pos].copy()
    pos["q50_oof"] = q50[mask_pos]
    pos["lx"] = np.log(pos["prev_month_spending"].values)
    pos["ly"] = np.log(pos["spending_30d"].values)
    pos["ratio"] = pos["spending_30d"].values / pos["prev_month_spending"].values

    # Distinct colorblind-safe colors per segment
    seg_meta = [
        (0, "Seg 0: Casual", ps.BLUE, 1.10),
        (1, "Seg 1: Grinder", ps.TEAL, 1.63),
        (2, "Seg 2: Social", ps.PURPLE, 1.82),
        (3, "Seg 3: Whale", ps.CORAL, 3.04),
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.28}
    )

    # --- Panel A: Stratified scatter (6,000 payers) + unit-slope fit lines ---
    rng = np.random.default_rng(42)
    n_per_seg = 1500
    xx_ref = np.linspace(1.3, 11.2, 200)
    ax1.plot(xx_ref, xx_ref, color=ps.GRAY, linestyle=":", linewidth=1.25, label="Identity y = prev (1.00x)")

    for seg_id, seg_name, col, base_ref in seg_meta:
        sub = pos[pos["segment"] == seg_id]
        lx = sub["lx"].values
        ly = sub["ly"].values
        r_pearson = float(np.corrcoef(lx, ly)[0, 1])
        med_ratio = float(np.median(sub["ratio"].values))
        slope, _ = np.polyfit(lx, ly, 1)

        idx = rng.choice(len(sub), size=min(n_per_seg, len(sub)), replace=False)
        ax1.scatter(
            lx[idx],
            ly[idx],
            s=8.0,
            alpha=0.22,
            color=col,
            edgecolors="none",
            rasterized=True,
        )

        # Unit-slope median offset line spanning [p01, p99] of segment's log(prev)
        x_lo, x_hi = np.quantile(lx, [0.01, 0.99])
        xs = np.array([x_lo, x_hi])
        ys = xs + np.log(med_ratio)
        ax1.plot(
            xs,
            ys,
            color=col,
            linewidth=2.2,
            label=f"{seg_name} (r={r_pearson:.3f}, base {base_ref:.2f}x, med {med_ratio:.2f}x)",
        )
        print(
            f"Seg {seg_id}: n={len(sub)}, r={r_pearson:.4f}, OLS_slope={slope:.3f}, "
            f"base_ref={base_ref:.2f}x, med_ratio={med_ratio:.2f}x"
        )

    ax1.set_xlabel(r"$\ln(\mathrm{prev\_month\_spending})$")
    ax1.set_ylabel(r"$\ln(\mathrm{spending\_30d})$")
    ax1.set_xlim(1.2, 11.4)
    ax1.set_ylim(1.0, 13.0)
    ax1.set_title("(a) Log-Linear Offset Structure (n = 6,000 Payers)", pad=10)
    # Place legend cleanly in empty lower-right triangle (y < prev)
    ax1.legend(loc="lower right", fontsize=6.9, frameon=True, handlelength=1.4)

    # Annotate Whale spikes cleanly in open upper-left space
    ax1.annotate(
        "Whale spikes\n(266 rows, 90k–300k)",
        xy=(7.15, 12.25),
        xytext=(2.15, 11.15),
        fontsize=7.8,
        color=ps.CORAL,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFF7ED", edgecolor=ps.CORAL, lw=0.8),
        arrowprops=dict(arrowstyle="->", color=ps.CORAL, lw=1.1, connectionstyle="arc3,rad=-0.12"),
    )

    # --- Panel B: Residual histogram for Segments 0-2 showing bounded trapezoid ---
    sub02 = pos[pos["segment"].isin([0, 1, 2])].copy()
    raw_lr = np.log(sub02["ratio"].values).copy()
    for s in [0, 1, 2]:
        mk = sub02["segment"].values == s
        raw_lr[mk] -= np.median(raw_lr[mk])

    oof_res = np.log(sub02["spending_30d"].values / sub02["q50_oof"].values)
    for s in [0, 1, 2]:
        mk = sub02["segment"].values == s
        oof_res[mk] -= np.median(oof_res[mk])

    sd_raw = float(np.std(raw_lr))
    sd_oof = float(np.std(oof_res))
    lo_bound, hi_bound = np.log(0.66), np.log(1.38)
    lo_plat, hi_plat = np.log(0.86), np.log(1.10)
    frac_in_bounds = float(np.mean((oof_res >= lo_bound) & (oof_res <= hi_bound)))
    print(f"Seg 0-2 residuals: sd_raw={sd_raw:.4f}, sd_oof={sd_oof:.4f}, in_[0.66,1.38]={frac_in_bounds:.4f}")

    bins = np.linspace(-0.56, 0.56, 29)
    ax2.hist(
        raw_lr,
        bins=bins,
        density=True,
        histtype="step",
        color=ps.SLATE,
        linewidth=1.35,
        linestyle="--",
        label=f"Raw seg-centered log(y/prev) (sd = {sd_raw:.3f})",
    )
    ax2.hist(
        oof_res,
        bins=bins,
        density=True,
        color=ps.TEAL,
        alpha=0.58,
        edgecolor=ps.NAVY,
        linewidth=0.65,
        label=f"OOF offset residual log(y/Q50) (sd = {sd_oof:.3f})",
    )

    a, b, c, d = 0.66, 0.86, 1.10, 1.38
    h_trap = 2.0 / ((d - a) + (c - b))
    zs = np.linspace(-0.52, 0.45, 500)
    us = np.exp(zs)
    fu = np.where(
        (us >= a) & (us < b),
        h_trap * (us - a) / (b - a),
        np.where((us >= b) & (us <= c), h_trap, np.where((us > c) & (us <= d), h_trap * (d - us) / (d - c), 0.0)),
    )
    fz = fu * us
    ax2.plot(zs, fz, color=ps.NAVY, linewidth=2.0, label="Theoretical trapezoid PDF [0.66, 1.38]")

    ax2.axvspan(lo_plat, hi_plat, color=ps.AMBER, alpha=0.14, label="Flat plateau [0.86x, 1.10x]")
    ax2.axvline(lo_bound, color=ps.CORAL, linestyle=":", linewidth=1.6)
    ax2.axvline(hi_bound, color=ps.CORAL, linestyle=":", linewidth=1.6)

    # Bound labels placed outside the density ramps in clean whitespace
    ax2.text(
        lo_bound - 0.015,
        1.55,
        f"0.66x\n({lo_bound:.2f})",
        ha="right",
        va="center",
        fontsize=7.6,
        color=ps.CORAL,
        fontweight="bold",
    )
    ax2.text(
        hi_bound + 0.015,
        1.55,
        f"1.38x\n(+{hi_bound:.2f})",
        ha="left",
        va="center",
        fontsize=7.6,
        color=ps.CORAL,
        fontweight="bold",
    )

    ax2.text(
        0.985,
        0.65,
        f"{frac_in_bounds*100:.1f}% within\n[0.66x, 1.38x]\nlog-sd = {sd_oof:.3f}",
        transform=ax2.transAxes,
        ha="right",
        va="center",
        fontsize=7.1,
        color=ps.NAVY,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.8),
    )

    ax2.set_xlabel(r"Centered Log Residual $\ln(y / \hat{y}_{\mathrm{med}})$")
    ax2.set_ylabel("Probability Density")
    ax2.set_xlim(-0.58, 0.62)
    ax2.set_ylim(0, 3.45)
    ax2.set_title("(b) Bounded Trapezoid Residual Floor (Seg 0–2)", pad=10)
    ax2.legend(loc="upper center", fontsize=7.0, frameon=True, ncol=1)

    out_path = os.path.join(OUT_DIR, "fig_t3_offset_linearity.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_fig3_model_progression_ceiling(facts):
    """Milestone progression bar chart vs power-law data ceiling & in-sample score."""
    ms = facts["task3"]["milestones"]

    labels = [
        "HGB\n2-Stage",
        "v4 XGB\nD_q",
        "v4 Honest\nNested Blend",
        "+ Offset\nlog(y/prev)\nXGB",
        "+ CatBoost\nd4 Offset",
        "+ 3x3\nSeed Bag",
        "Final 10x10\nSeed Bag",
    ]
    scores = [
        ms["hgb_two_stage"],
        ms["v4_xgb_dq"],
        ms["v4_nested_lad_honest"],
        ms["xgb_offset_prev"],
        ms["cb_d4_offset_single"],
        ms["cb_d4_3seed_bag"],
        ms["cb_d4_10x10_bag_final"],
    ]
    colors = [
        ps.GRAY,
        ps.SLATE,
        ps.BLUE,
        ps.TEAL,
        ps.TEAL,
        ps.NAVY,
        ps.NAVY,
    ]
    insample = ms["insample_train_score"]
    ceil_lo, ceil_hi = 0.7960, 0.7970

    fig, ax = plt.subplots(figsize=(9.8, 4.4))

    x = np.arange(len(labels))
    bars = ax.bar(x, scores, width=0.56, color=colors, alpha=0.92, zorder=3)

    # Shaded power-law data ceiling band
    ax.axhspan(ceil_lo, ceil_hi, color=ps.AMBER, alpha=0.22, zorder=2)
    ax.axhline(ceil_lo, color=ps.AMBER, linestyle=":", linewidth=1.0, alpha=0.75, zorder=2)
    ax.axhline(ceil_hi, color=ps.AMBER, linestyle=":", linewidth=1.0, alpha=0.75, zorder=2)
    ax.text(
        0.04,
        0.79735,
        "Power-Law Data Ceiling (~0.796–0.797)",
        ha="left",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color="#92400E",
    )

    # In-sample train score dashed line
    ax.axhline(insample, color=ps.CORAL, linestyle="--", linewidth=1.6, zorder=4)
    ax.text(
        0.04,
        insample + 0.00038,
        f"In-Sample Train Score ({insample:.5f}) — target noise ceiling, not underfit",
        ha="left",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color=ps.CORAL,
    )

    # Value labels above each bar
    for i, (bar, sc) in enumerate(zip(bars, scores)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            sc + 0.00016,
            f"{sc:.5f}",
            ha="center",
            va="bottom",
            fontsize=7.7,
            fontweight="bold" if i in (0, 3, 4, 6) else "normal",
            color="#1E293B",
        )

    ax.annotate(
        f"+{scores[3]-scores[2]:.5f}\n(offset target)",
        xy=(3, scores[3] + 0.0012),
        xytext=(2.35, 0.7991),
        ha="center",
        fontsize=7.7,
        color=ps.TEAL,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ps.TEAL, lw=1.05),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.3)
    ax.set_ylabel("5-Fold OOF NMAE (Higher is Better)")
    ax.set_ylim(0.7810, 0.8062)
    ax.grid(axis="x", visible=False)
    ax.set_title("Task 3 OOF NMAE Progression vs. Extrapolated Data Ceiling", pad=22)
    ax.text(
        0.5,
        1.015,
        "All-Zero Baseline NMAE = 0.50000  (y-axis zoomed to [0.781, 0.806] to show incremental deltas)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="#475569",
    )

    out_path = os.path.join(OUT_DIR, "fig_t3_model_progression_ceiling.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ps.apply_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    facts = load_facts()
    make_fig1_whale_and_error_budget(facts)
    make_fig2_offset_linearity()
    make_fig3_model_progression_ceiling(facts)


if __name__ == "__main__":
    main()
