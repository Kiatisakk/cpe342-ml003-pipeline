"""Task 5 - visualise every anomaly type (writes PNGs to t123/research/t5/figs/).

Usage (from repo root, after t5_pipeline.py and t5_box_analysis.py):
    python t123/research/t5/t5_plot_clusters.py

fig1_types.png      one panel per injected type: the two defining metrics for ONE period
                    (stationary types: period 2; shock types: period 4). Normal rows grey, the type coloured,
                    its box dashed. A uniform-in-a-box type shows up as a filled rectangle.
fig2_new_type.png   the new type: 4-period means (how it was found) + per-period values (why it is exact).
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SRC = "cpe342-kaggle-main/05_anomaly_detection/test.csv"
OUT = "t123/research/t5/figs"; os.makedirs(OUT, exist_ok=True)
X = pd.read_csv(SRC); G = pd.read_csv("submissions/probes_t5/t5_groups.csv")
S = pd.read_csv("t123/research/t5/out/t5_pipeline_scores.csv")
P = [1, 2, 3, 4]; u = G.iloc[:, 1:].max(axis=1).values == 1
new = (S["C0"].values == 1) & ~u
rule = G.rule.values == 1

def ext(c):
    v = X[c].values; lo, hi = np.nanquantile(v[~u], [.0005, .9995]); return (v < lo) | (v > hi)
take = rule & (ext("currency_transfers_4") | ext("play_pattern_consistency_4"))
sell = rule & ~take & (ext("skill_trajectory_4") | ext("skill_rating_4") | ext("friend_removes_4"))
vpn = rule & ~take & ~sell & (ext("purchase_frequency_4") | ext("purchase_amount_mean_4"))

# palette (reference instance, validated all-pairs for 3 slots on the light surface)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NORMAL, BLUE, ORANGE, AQUA = "#c9c8c2", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(0)
bg = np.where(~u & ~new)[0]; bg = rng.choice(bg, 6000, replace=False)   # thin the 22k normal rows

def panel(ax, xcol, ycol, mask, color, title, n, box=None, logx=False, logy=False, label="type"):
    x, y = X[xcol].values, X[ycol].values
    ax.scatter(x[bg], y[bg], s=3, c=NORMAL, lw=0, alpha=.6, rasterized=True)
    ax.scatter(x[mask], y[mask], s=9, c=color, lw=.4, edgecolors=SURF, rasterized=True)
    if box:
        (x0, x1), (y0, y1) = box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=1, ec=INK2))
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_title(f"{title}  ({n} rows)", loc="left")
    ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.grid(color=GRID, lw=.5); ax.set_axisbelow(True)

# ---------------- figure 1: all eight types ----------------
fig, axs = plt.subplots(3, 3, figsize=(15, 13.5))
spec = [
    ("accuracy_std_2", "kda_std_2", new, ORANGE, "NEW acckda", ((9, 12), (.7, 1.0)), {}),
    ("ip_hash_entropy_2", "login_time_variance_2", G.bot.values == 1, BLUE, "bot", ((0, .2), (.1, .5)), {}),
    ("activity_level_2", "skill_rating_2", G.lowact.values == 1, BLUE, "lowact", ((.1, .3), (2200, 2500)), {}),
    ("communication_frequency_2", "friend_adds_2", G.lowcomm.values == 1, BLUE, "lowcomm", ((0, .1), (15, 25)), {}),
    ("behavior_stability", "purchase_frequency_2", G.lowstab.values == 1, BLUE, "lowstab", ((.2, .4), (3, 6)), {}),
    ("currency_transfers_4", "play_pattern_consistency_4", take, AQUA, "takeover (period 4)", None, {}),
    ("skill_rating_4", "friend_removes_4", sell, AQUA, "selling (period 4)", None, {}),
    ("location_changes_4", "purchase_frequency_4", vpn, AQUA, "VPN (period 4)", None, {}),
]
for ax, (xc, yc, m, col, t, box, kw) in zip(axs.ravel(), spec):
    panel(ax, xc, yc, m, col, t, int(m.sum()), box, **kw)
# last panel: the counts
ax = axs.ravel()[-1]
names = ["takeover", "selling", "VPN", "lowstab", "lowact", "NEW acckda", "lowcomm", "bot"]
vals = [take.sum(), sell.sum(), vpn.sum(), (G.lowstab == 1).sum(), (G.lowact == 1).sum(), new.sum(), (G.lowcomm == 1).sum(), (G.bot == 1).sum()]
cols = [AQUA] * 3 + [BLUE] * 2 + [ORANGE] + [BLUE] * 2
ax.barh(names[::-1], vals[::-1], color=cols[::-1], height=.62)
for i, v in enumerate(vals[::-1]): ax.text(v + 6, i, str(v), va="center", color=INK2)
ax.set_xlim(0, 380); ax.set_title("rows per type (total 2,055)", loc="left"); ax.grid(axis="x", color=GRID, lw=.5); ax.set_axisbelow(True)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
h = [plt.Line2D([], [], marker="o", ls="", ms=6, mfc=c, mec=c, label=l) for c, l in
     [(NORMAL, "normal rows (random 6,000 of 23,834)"), (BLUE, "stationary type in the current union"),
      (AQUA, "period-4 shock type in the current union"), (ORANGE, "NEW type (not in the union)")]] + \
    [plt.Line2D([], [], ls="--", color=INK2, label="box the type is drawn from")]
fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5, .965))
fig.suptitle("T5 — every injected anomaly type, one period of its two defining metrics", x=.01, ha="left", y=.99, fontsize=13, weight="bold")
fig.tight_layout(rect=(0, 0, 1, .945)); fig.savefig(f"{OUT}/fig1_types.png", dpi=160); plt.close(fig)

# ---------------- figure 2: the new type in detail ----------------
mean = lambda b: X[[f"{b}_{p}" for p in P]].mean(axis=1).values
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
A, K, PS = mean("accuracy_std"), mean("kda_std"), mean("performance_stability")
ax = axs[0]
ax.scatter(A[bg], K[bg], s=3, c=NORMAL, lw=0, alpha=.6, rasterized=True)
ax.scatter(A[new], K[new], s=10, c=ORANGE, lw=.4, edgecolors=SURF)
ax.set_xlabel("accuracy_std (mean of 4 periods)"); ax.set_ylabel("kda_std (mean of 4 periods)")
ax.set_title("1  How it was found: a dense corner in the means", loc="left"); ax.grid(color=GRID, lw=.5); ax.set_axisbelow(True)
ax = axs[1]
for p, lab in [(2, None)]:
    xs, ys = X[f"performance_stability_{p}"].values, X[f"kda_std_{p}"].values
ax.scatter(xs[bg], ys[bg], s=3, c=NORMAL, lw=0, alpha=.6, rasterized=True)
ax.scatter(xs[new], ys[new], s=10, c=ORANGE, lw=.4, edgecolors=SURF)
ax.add_patch(Rectangle((.85, .7), .1, .3, fill=False, ls="--", lw=1, ec=INK2))
ax.set_xlabel("performance_stability (period 2)"); ax.set_ylabel("kda_std (period 2)")
ax.set_title("2  Why it is exact: per-period values fill a box", loc="left"); ax.grid(color=GRID, lw=.5); ax.set_axisbelow(True)
ax = axs[2]
v_new = X.loc[new, [f"performance_stability_{p}" for p in P]].values.ravel(); v_new = v_new[~np.isnan(v_new)]
v_nor = X.loc[~u & ~new, [f"performance_stability_{p}" for p in P]].values.ravel(); v_nor = v_nor[~np.isnan(v_nor)]
bins = np.linspace(.55, 1.05, 101)
ax.hist(v_nor, bins=bins, color=NORMAL, weights=np.full(len(v_nor), 1 / 100))
ax.hist(v_new, bins=bins, color=ORANGE)
for e in (.85, .95): ax.axvline(e, ls="--", lw=1, color=INK2)
ax.text(.965, ax.get_ylim()[1] * .9, "members:\nuniform 0.85–0.95", ha="left", va="top", color=INK2)
ax.set_xlabel("performance_stability, all periods"); ax.set_ylabel("count (normal ÷ 100)")
ax.set_title("3  Hard edges: no member value outside the box", loc="left"); ax.grid(axis="y", color=GRID, lw=.5); ax.set_axisbelow(True)
fig.suptitle(f"New type 'acckda': {new.sum()} rows, 0 in the current union, ~1 expected by chance", x=.01, ha="left", fontsize=13, weight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_new_type.png", dpi=160); plt.close(fig)
print("wrote", OUT)
