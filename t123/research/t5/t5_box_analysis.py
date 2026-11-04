"""Task 5 - "box" analyses (round 2 of the research; numbers in REPORT.md section "Round 2").

Usage (from repo root):
    python t123/research/t5/t5_box_analysis.py [test.csv] [submissions/probes_t5/t5_groups.csv]

Finding: every injected stationary type replaces some metrics, in every period independently, by a uniform
draw inside a fixed box with round bounds. A row of a type therefore has ALL its observed periods inside
the box; a normal row (stable personal base + period noise) almost never does for several metrics at once.

  1. Box definitions of the known types and of the new type; exact reproduction of the union groups.
  2. New type: per-period bounds, bound sensitivity, and the chance rate of a normal row falling inside
     (null = per-metric distribution of "#periods outside the box" in normal rows, convolved over metrics,
     which respects the within-metric correlation across periods).
  3. Box-pair search: for every metric and a grid of intervals, I = "all 4 periods inside [a, b]";
     co-occurrence of two indicators of different metrics vs independence, outside union + new box.
  4. Shock group (rule) = 3 subtypes at period 4; joint-signature score of non-union rows.
"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
SRC = sys.argv[1] if len(sys.argv) > 1 else "cpe342-kaggle-main/05_anomaly_detection/test.csv"
GRP = sys.argv[2] if len(sys.argv) > 2 else "submissions/probes_t5/t5_groups.csv"
X = pd.read_csv(SRC); G = pd.read_csv(GRP); P = [1, 2, 3, 4]
u = G.iloc[:, 1:].max(axis=1).values == 1
H = lambda s: print("\n" + "=" * 8, s, "=" * 8)
STATIC = ["behavior_stability", "account_age_days"]
cols_of = lambda b: [b] if b in STATIC else [f"{b}_{p}" for p in P]
def inbox(b, lo, hi):
    m = X[cols_of(b)].values.astype(float)
    return (((m >= lo) & (m <= hi)) | np.isnan(m)).all(1)
def nviol(b, lo, hi):
    m = X[cols_of(b)].values.astype(float)
    return (~(((m >= lo) & (m <= hi)) | np.isnan(m))).sum(1)

BOX = {"bot": {"ip_hash_entropy": (0, .2), "kda_std": (.05, .15), "login_time_variance": (.1, .5), "performance_stability": (.95, .99),
               "play_pattern_consistency": (.95, .99), "session_duration_std": (1, 5), "device_count": (1, 1)},
       "lowact": {"activity_level": (.1, .3), "purchase_amount_mean": (300, 500), "skill_rating": (2200, 2500)},
       "lowcomm": {"communication_frequency": (0, .1), "friend_adds": (15, 25), "guild_changes": (2, 3)},
       "lowstab": {"behavior_stability": (.2, .4), "account_age_days": (800, 1500), "purchase_frequency": (3, 6)},
       "acckda (NEW)": {"performance_stability": (.85, .95), "kda_std": (.7, 1.0), "accuracy_std": (9, 12)}}

H("1. box definitions reproduce the union groups")
boxes = {}
for g, bx in BOX.items():
    ok = np.ones(len(X), bool)
    for b, (lo, hi) in bx.items(): ok &= inbox(b, lo, hi)
    boxes[g] = ok
    if g in G: ref = G[g].values == 1; print(f"{g:13s} box {ok.sum():4d} | group {ref.sum():4d} | both {(ok & ref).sum():4d} | box-only {(ok & ~ref).sum()} | group-only {(ref & ~ok).sum()}   {bx}")
    else: print(f"{g:13s} box {ok.sum():4d} | inside union {(ok & u).sum()}   {bx}")
new = boxes["acckda (NEW)"] & ~u

H("2. new type: bounds, sensitivity, chance rate")
for b in BOX["acckda (NEW)"]:
    v = X.loc[new, cols_of(b)].values.ravel(); v = v[~np.isnan(v)]
    w = X.loc[~u & ~new, cols_of(b)].values.ravel(); w = w[~np.isnan(w)]
    print(f"{b:22s} members per-period [{v.min():.4f}, {v.max():.4f}] ({len(v)} values) | normal 0.1-99.9% [{np.quantile(w, .001):.3f}, {np.quantile(w, .999):.3f}]")
def count(bx):
    ok = ~u.copy()
    for b, (lo, hi) in bx.items(): ok &= inbox(b, lo, hi)
    return int(ok.sum())
base = BOX["acckda (NEW)"]
for b, (lo, hi) in base.items():
    w = hi - lo; res = []
    for dlo, dhi in [(-.1, 0), (0, .1), (.1, 0), (0, -.1)]:
        t = dict(base); t[b] = (lo + dlo * w, hi + dhi * w); res.append(count(t))
    print(f"{b:22s} widen low/high by 10%: {res[0]}, {res[1]} | tighten low/high by 10%: {res[2]}, {res[3]}  (exact box: {count(base)})")
V = pd.DataFrame({b: nviol(b, lo, hi) for b, (lo, hi) in base.items()})
ref = ~u & ~new; nd = np.array([1.0])
for b in base: nd = np.convolve(nd, np.bincount(V.loc[ref, b], minlength=5) / ref.sum())
obs = np.bincount(V.sum(axis=1)[~u], minlength=13)
for k in range(4): print(f"rows with {k} out-of-box period values: observed {obs[k]:4d} | null expected {nd[k] * (~u).sum():7.2f}")

H("3. box-pair search outside union + new box (calibration with all rows first)")
mets = sorted({c[:-2] for c in X.columns if c[-2:] in ("_1", "_2", "_3", "_4")} - {"login_lat", "login_lon"})
def boxpair(rows, K=13, top=5):
    names, cols = [], []
    for b in mets + STATIC:
        m = X.loc[rows, cols_of(b)].values.astype(float); lo, hi = np.nanquantile(m, [.001, .999]); g = np.linspace(lo, hi, K)
        for i in range(K):
            for j in range(i + 1, min(K, i + K // 2 + 1)):
                a = g[i] if i > 0 else -np.inf; z = g[j] if j < K - 1 else np.inf
                f = (((m >= a) & (m <= z)) | np.isnan(m)).all(1)
                if 30 < f.sum() < .5 * rows.sum(): names.append((b, round(g[i], 3), round(g[j], 3))); cols.append(f)
    T = np.stack(cols, 1).astype(np.float32); N = len(T); C = T.T @ T; n = T.sum(0); E = np.outer(n, n) / N
    Z = (C - E) / np.sqrt(E + 1); met = np.array([x[0] for x in names]); Z[met[:, None] == met[None, :]] = 0
    iu = np.triu_indices(len(names), 1); zz = Z[iu]; o = np.argsort(-zz)
    print(f"  {len(names)} indicators, {len(zz)} pairs; top pairs:")
    seen = set(); k = 0
    for q in o:
        i, j = iu[0][q], iu[1][q]; key = frozenset([names[i][0], names[j][0]])
        if key in seen: continue
        seen.add(key); print(f"    z={zz[q]:6.1f} {names[i]} x {names[j]} obs {int(C[i, j])} exp {E[i, j]:.1f}"); k += 1
        if k >= top: break
print("all rows:"); boxpair(np.ones(len(X), bool), top=3)
print("outside union and new box:"); boxpair(~u & ~new)

H("4. shock subtypes (period 4) and joint-signature check outside")
r = G.rule.values == 1
def ext(c):
    v = X[c].values; lo, hi = np.nanquantile(v[~u], [.0005, .9995]); return (v < lo) | (v > hi)
t1 = r & (ext("currency_transfers_4") | ext("play_pattern_consistency_4"))
t3 = r & ~t1 & (ext("skill_trajectory_4") | ext("skill_rating_4") | ext("friend_removes_4"))
t2 = r & ~t1 & ~t3 & (ext("purchase_frequency_4") | ext("purchase_amount_mean_4"))
print(f"takeover {t1.sum()} | selling {t3.sum()} | vpn {t2.sum()} | other {(r & ~t1 & ~t2 & ~t3).sum()}")
def ratio(b):
    m = X[[f"{b}_{p}" for p in P]].values; return m[:, 3] / np.nanmean(m[:, :3], 1)
def diff(b):
    m = X[[f"{b}_{p}" for p in P]].values; return m[:, 3] - np.nanmean(m[:, :3], 1)
SIG = {"takeover": (t1, [("play_pattern_consistency", -1), ("currency_transfers", 1), ("location_changes", 1), ("device_count", 1)]),
       "selling": (t3, [("skill_rating", 1), ("kda_mean", 1), ("accuracy_mean", 1), ("reaction_time_mean", -1), ("friend_removes", 1), ("skill_trajectory", 0)]),
       "vpn": (t2, [("purchase_amount_mean", 1), ("purchase_frequency", 1), ("location_changes", 1)])}
out = ~u & ~new
for nm, (s, lst) in SIG.items():
    hits = []
    for b, sg in lst:
        x = diff(b) if sg == 0 else ratio(b); thr = np.nanquantile(x[s], .05 if sg >= 0 else .95)
        hits.append(x > thr if sg >= 0 else x < thr)
    k = np.stack(hits, 1).sum(1)
    print(f"{nm:9s} #signature metrics beyond the members' 5% level -> members {np.bincount(k[s], minlength=len(lst) + 1).tolist()} | outside {np.bincount(k[out], minlength=len(lst) + 1).tolist()}")
print("\nType sizes: takeover %d, selling %d, vpn %d (+%d geo-only), bot %d, lowact %d, lowcomm %d, lowstab %d, NEW acckda %d -> K = %d"
      % (t1.sum(), t3.sum(), t2.sum(), (r & ~t1 & ~t2 & ~t3).sum(), boxes["bot"].sum(), boxes["lowact"].sum(), boxes["lowcomm"].sum(),
         (G.lowstab == 1).sum(), new.sum(), u.sum() + new.sum()))

H("5. lowstab members hidden by NaN behavior_stability (the union rule is a single static column)")
ls = G.lowstab.values == 1; bsn = X.behavior_stability.isna().values
pf = inbox("purchase_frequency", 3, 6); age = inbox("account_age_days", 800, 1500)
nr = ~u & ~bsn
cand = bsn & ~u & pf & age
exp_chance = (pf & age)[nr].mean() * (bsn & ~u).sum()
print(f"NaN behavior_stability rows {bsn.sum()} (in union {(bsn & u).sum()}); expected hidden members ~ {ls.sum() * bsn.mean() / (1 - bsn.mean()):.0f}")
print(f"NaN-bs non-union rows inside purchase_freq[3,6] & age[800,1500]: observed {cand.sum()} vs chance {exp_chance:.1f} -> excess {cand.sum() - exp_chance:.0f} +- {np.sqrt(exp_chance):.0f}")
lm = X[[f"login_count_{p}" for p in P]].mean(axis=1).values; ps = X[[f"purchase_frequency_{p}" for p in P]].std(axis=1).values
nor = nr & pf & age & (X.behavior_stability.values >= 0.6)
def logpdf(v, ref, bins):
    h, _ = np.histogram(ref[~np.isnan(ref)], bins=bins); h = (h + .5) / (h.sum() + .5 * len(h)) / np.diff(bins)
    o = np.log(h[np.clip(np.digitize(v, bins) - 1, 0, len(h) - 1)]); o[np.isnan(v)] = 0; return o
bl, bp = np.linspace(0, 130, 27), np.linspace(0, 2.5, 26)
LLR = logpdf(lm, lm[ls], bl) - logpdf(lm, lm[nor], bl) + logpdf(ps, ps[ls], bp) - logpdf(ps, ps[nor], bp)
prior = 14 / cand.sum(); post = np.where(cand, 1 / (1 + np.exp(-(LLR + np.log(prior / (1 - prior))))), 0)
print("members login_count mean q50 %.1f vs normal-in-box %.1f; purchase_freq period-sd q50 %.2f vs %.2f"
      % (np.nanmedian(lm[ls]), np.nanmedian(lm[nor]), np.nanmedian(ps[ls]), np.nanmedian(ps[nor])))
for t in [.5, .2, .1]: s = post > t; print(f"  posterior > {t}: {s.sum()} rows, expected true {post[s].sum():.1f}")
pd.DataFrame({"id": [f"ANS{i + 1:05d}" for i in range(len(X))], "post_lowstab_nan": post.round(4)}).to_csv(
    "t123/research/t5/out/lowstab_nan_scores.csv", index=False)
