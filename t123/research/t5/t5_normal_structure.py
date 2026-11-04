"""Task 5 - is there any structure left among the rows the final set calls normal?

Usage (from repo root, after t5_pipeline.py and t5_box_analysis.py):
    python t123/research/t5/t5_normal_structure.py
1. 4-feature grid scan (5 bins^4 rank cells, every 4-feature combination of 28 4-period means/statics) on
   the normal rows vs 6 column-shuffled nulls (same marginals, no dependence); calibration with the new
   type put back.
2. t-SNE map (figs/fig3_tsne.png): 6,000 normal rows + every flagged row; left = real, right = the same
   normal rows column-shuffled. Only types that differ in many features become islands; 2-3-feature types
   (incl. the new one) do not, so a smooth blob is NOT evidence of no clusters - the low-dimensional scans are.
Log: out/normal_structure_log.txt
"""
import itertools, numpy as np, pandas as pd, matplotlib, warnings
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.manifold import TSNE
warnings.filterwarnings("ignore")
X = pd.read_csv("cpe342-kaggle-main/05_anomaly_detection/test.csv"); S = pd.read_csv("t123/research/t5/out/t5_pipeline_scores.csv")
G = pd.read_csv("submissions/probes_t5/t5_groups.csv"); P = [1, 2, 3, 4]
mets = sorted({c[:-2] for c in X.columns if c[-2:] in ("_1", "_2", "_3", "_4")} - {"login_lat", "login_lon", "device_count", "refund_count", "guild_changes"})
F = pd.DataFrame({b: X[[f"{b}_{p}" for p in P]].mean(axis=1) for b in mets}); F["account_age_days"] = X.account_age_days; F["behavior_stability"] = X.behavior_stability
norm = S.final.values == 0; new = S.C0.values == 1
log = open("t123/research/t5/out/normal_structure_log.txt", "w")
def say(*a): print(*a); print(*a, file=log)

def scan(D, B=5):
    R = D.rank(pct=True, method="first").values; ok = ~D.isna().values
    Q = np.minimum((np.nan_to_num(R) * B).astype(int), B - 1); out = []
    for c in itertools.combinations(range(D.shape[1]), 4):
        v = ok[:, c].all(1); idx = ((Q[v, c[0]] * B + Q[v, c[1]]) * B + Q[v, c[2]]) * B + Q[v, c[3]]
        cnt = np.bincount(idx, minlength=B ** 4); e = v.sum() / B ** 4
        out.append(((cnt.max() - e) / np.sqrt(e), [D.columns[i] for i in c], int(cnt.max()), round(e, 1)))
    return sorted(out, key=lambda t: -t[0])
def shuf(D, s):   # independent permutation of every column (breaks dependence, keeps marginals)
    g = np.random.default_rng(s); return pd.DataFrame({c: g.permutation(D[c].values) for c in D.columns})
Fn = F[norm].reset_index(drop=True)
real = scan(Fn); nulls = [scan(shuf(Fn, s))[0][0] for s in range(1, 7)]
say(f"1. 4-feature scan, {len(real)} combinations on {len(Fn)} normal rows")
say(f"   real top z = {real[0][0]:.2f} {real[0][1]} ({real[0][2]} rows vs {real[0][3]} expected)")
say(f"   shuffled-null top z over 6 reps: {np.round(nulls, 2).tolist()}  -> real is inside the null range" if real[0][0] <= max(nulls) else "   REAL ABOVE ALL NULLS")
cal = scan(F[norm | new].reset_index(drop=True))
say(f"   calibration (normal + {new.sum()} new-type rows): top z = {cal[0][0]:.2f} {cal[0][1]}")

# ---- t-SNE ----
rng = np.random.default_rng(0)
idx_n = rng.choice(np.where(norm)[0], 6000, replace=False); idx_a = np.where(~norm)[0]
lab = np.full(len(X), "normal", object)
for g in ["bot", "lowact", "lowcomm", "lowstab"]: lab[G[g].values == 1] = g
t1 = pd.read_csv("t123/research/t5/out/t5_pipeline_scores.csv")
lab[new] = "NEW acckda"
r = G.rule.values == 1
def ext(c):
    v = X[c].values; lo, hi = np.nanquantile(v[norm], [.0005, .9995]); return (v < lo) | (v > hi)
tk = r & (ext("currency_transfers_4") | ext("play_pattern_consistency_4")); sl = r & ~tk & (ext("skill_trajectory_4") | ext("skill_rating_4") | ext("friend_removes_4"))
lab[r] = "VPN"; lab[tk] = "takeover"; lab[sl] = "selling"
# features: 4-period means + statics + period-4 change of every metric (so shock types are visible too)
D4 = pd.DataFrame({f"{b}@4": X[f"{b}_4"] - X[[f"{b}_{p}" for p in [1, 2, 3]]].mean(axis=1) for b in mets})
FF = pd.concat([F, D4], axis=1)
def embed(A):
    A = A.fillna(A.median()).values + rng.normal(0, 1e-6, A.shape)          # jitter breaks ties at 0
    Z = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0).fit_transform(A)
    return TSNE(2, perplexity=50, init="pca", random_state=0).fit_transform(Z)
ids = np.r_[idx_n, idx_a]
E = embed(FF.iloc[ids]); L = lab[ids]
En = embed(shuf(FF.iloc[idx_n].reset_index(drop=True), 9))

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NORMAL, BLUE, ORANGE = "#c9c8c2", "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "font.size": 9,
                     "axes.titlesize": 10.5, "axes.titleweight": "bold"})
fig, axs = plt.subplots(1, 2, figsize=(15, 7.2))
ax = axs[0]
m = L == "normal"; ax.scatter(E[m, 0], E[m, 1], s=3, c=NORMAL, lw=0, rasterized=True)
for t in ["bot", "lowact", "lowcomm", "lowstab", "takeover", "selling", "VPN", "NEW acckda"]:
    m = L == t; col = ORANGE if t.startswith("NEW") else BLUE
    ax.scatter(E[m, 0], E[m, 1], s=7, c=col, lw=.3, edgecolors=SURF, rasterized=True)
    cx, cy = np.median(E[m, 0]), np.median(E[m, 1])
    ax.annotate(f"{t} ({m.sum()})", (cx, cy), xytext=(0, 14), textcoords="offset points", ha="center", color=INK, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=.85))
ax.set_title("Real: 6,000 normal rows (grey) + all 2,055 flagged rows", loc="left")
ax = axs[1]
ax.scatter(En[:, 0], En[:, 1], s=3, c=NORMAL, lw=0, rasterized=True)
ax.set_title("Null: the same 6,000 normal rows, each column shuffled", loc="left")
for a in axs: a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
h = [plt.Line2D([], [], marker="o", ls="", ms=6, mfc=c, mec=c, label=l) for c, l in [(NORMAL, "normal (final set = 0)"), (BLUE, "type in current union"), (ORANGE, "NEW type")]]
fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .955))
fig.suptitle("t-SNE (56 features): bots and the 3 shock types form islands, but types defined by only 2-3 features (lowact, lowcomm, lowstab, NEW) dissolve into the blob",
             x=.01, ha="left", y=.995, fontsize=12, weight="bold")
fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig("t123/research/t5/figs/fig3_tsne.png", dpi=160)
say("2. t-SNE written to figs/fig3_tsne.png")
