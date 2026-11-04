"""Task 5 - unsupervised anomaly pipeline (model-based reproduction of the union + new groups).

Usage (from repo root):
    python t123/research/t5/t5_pipeline.py [cpe342-kaggle-main/05_anomaly_detection/test.csv] [out_dir]

Stages
  A. Temporal shock detector (robust z, median/MAD): change of periods 3/4 vs the mean of periods 1/2
     for every metric + geo jump of the login location.  Flags account takeover / selling / geo-VPN.
  B0. 1-D density-gap detector: per feature (4-period mean / static), histogram over the range; a run of
     (near-)empty bins that cuts off a small side (50..2000 rows) = isolated stationary cluster
     (bots, low activity, low communication, low behaviour stability).
  B. Dependency discovery: in normal data every pair of features is independent (|r| < 0.08).
     Among rows not flagged by A, count how often two features are jointly in their 5% tails and
     compare with the independence expectation.  Significant pairs form a graph; each connected
     component is a candidate injected type (feature subset + direction).
  C. Mixture model per component: EM on [Gaussian cluster] vs [background = product of 1-D marginal
     KDEs, i.e. the independence null].  Posterior P(cluster | x) is the anomaly score; a row is
     flagged when the posterior exceeds TAU (F3-optimal: marginal precision ~ F3/10 ~ 0.09).
     After a first EM pass the component is expanded with every feature whose cluster mean differs from
     the background by > 0.6 sd (signature expansion; next-largest shift is ~0.3 sd), and EM is run again;
     from the second round the pool also holds the level-adjusted period-to-period variability ("~vol")
     of the signature features (the new cluster has ~half the normal period noise in performance_stability).
  C2. Support estimation: the box (per-period min/max) of each signature metric is estimated from the
     confident members (posterior > 0.9); every row whose observed periods all fall inside is flagged.
  D. Isolation Forest on the same features (reported as an independent check, soft vote).
Final = A OR B0 OR any C2 box (hard OR-vote).
"""
import sys, os, json, numpy as np, pandas as pd, warnings
from scipy.stats import multivariate_normal
from sklearn.ensemble import IsolationForest
warnings.filterwarnings("ignore")

SRC = sys.argv[1] if len(sys.argv) > 1 else "cpe342-kaggle-main/05_anomaly_detection/test.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "t123/research/t5/out"
os.makedirs(OUT, exist_ok=True)
TAU = 0.10
EXP_T = float(os.environ.get("T5_EXP_T", 0.6))   # signature-expansion threshold in sd
X = pd.read_csv(SRC); P = [1, 2, 3, 4]; ID = X.columns[0]
ALLB = sorted({c[:-2] for c in X.columns if c[-2:] in ("_1", "_2", "_3", "_4")})

# ---------------- A. temporal shock detector ----------------
lat = np.stack([X[f"login_lat_{p}"] for p in P], 1).astype(float)
lon = np.stack([X[f"login_lon_{p}"] for p in P], 1).astype(float)
bad_lat = (np.abs(lat) > 90).any(1)
lt = np.where(np.abs(lat) > 90, np.nan, lat)
hl, hn = np.nanmean(lt[:, :2], 1), np.nanmean(lon[:, :2], 1)
jump = np.nan_to_num(np.nanmax(np.hypot(lt[:, 2:] - hl[:, None], (lon[:, 2:] - hn[:, None] + 180) % 360 - 180), 1))
Z = []
for b in [b for b in ALLB if b not in ("login_lat", "login_lon")]:
    m = X[[f"{b}_{p}" for p in P]].astype(float).values
    ch = np.nanmax(np.abs(m[:, 2:] - np.nanmean(m[:, :2], 1, keepdims=True)), 1)
    med = np.nanmedian(ch); mad = 1.4826 * np.nanmedian(np.abs(ch - med)) + 1e-9
    Z.append((ch - med) / mad)
Z = np.nan_to_num(np.stack(Z, 1))
z2 = np.sort(Z, 1)[:, -2]
shock = (jump > 5) | bad_lat | (z2 > 6)
shock_score = np.maximum(z2 / 6, jump / 5)

# ---------------- B. dependency discovery on per-row 4-period means ----------------
CONT = [b for b in ALLB if b not in ("login_lat", "login_lon", "device_count", "refund_count", "guild_changes")]
F = pd.DataFrame({b: X[[f"{b}_{p}" for p in P]].mean(axis=1) for b in CONT})
F["account_age_days"] = X.account_age_days; F["behavior_stability"] = X.behavior_stability

def gap_clusters(v, nb=80, lowfrac=0.02, minrun=3):
    ok = ~np.isnan(v); h, e = np.histogram(v[ok], bins=nb)
    plate = np.median(h[h > 0]); low = h <= lowfrac * plate
    out = []; i = 0
    while i < nb:
        if low[i]:
            j = i
            while j < nb and low[j]: j += 1
            if j - i >= minrun and i > 0 and j < nb:
                nl, nr = h[:i].sum(), h[j:].sum()
                if 50 <= min(nl, nr) <= 2000:
                    cut = (e[i] + e[j]) / 2
                    out.append((cut, -1 if nl < nr else +1, int(min(nl, nr))))
            i = j
        else: i += 1
    return out

gap = {}
for c in F.columns:
    v = F[c].values.astype(float); v_ = np.where(shock, np.nan, v)
    for cut, side, n in gap_clusters(v_):
        flag = (~shock) & ((v < cut) if side < 0 else (v > cut))
        gap[f"{c}{'<' if side < 0 else '>'}{cut:.3f}"] = flag
# merge gap clusters that share > 50% of rows (same injected type seen through several features)
groups = []
for k, f in sorted(gap.items(), key=lambda t: -t[1].sum()):
    for g in groups:
        if (f & g["flag"]).sum() > 0.5 * min(f.sum(), g["flag"].sum()):
            g["rules"].append(k); g["flag"] = g["flag"] | f; break
    else: groups.append({"rules": [k], "flag": f.copy()})
print("B0 gap clusters:")
for g in groups: print(f"   n={int(g['flag'].sum()):4d}  {g['rules']}")
gapflag = np.zeros(len(X), bool)
for g in groups: gapflag |= g["flag"]
rest = ~shock & ~gapflag
Fr = F[rest]
q = 0.05
T = {}
for c in F.columns:
    T[(c, -1)] = (Fr[c] < Fr[c].quantile(q)).values
    T[(c, +1)] = (Fr[c] > Fr[c].quantile(1 - q)).values
keys = list(T); Tm = np.stack([T[k] for k in keys], 1).astype(float)
cnt = Tm.T @ Tm; pr = Tm.mean(0); E = np.outer(pr, pr) * len(Tm); Zp = (cnt - E) / np.sqrt(E)
ZMIN = 6.0
edges = [(keys[i], keys[j], Zp[i, j]) for i in range(len(keys)) for j in range(i + 1, len(keys))
         if keys[i][0] != keys[j][0] and Zp[i, j] > ZMIN]
# connected components
parent = {}
def find(a):
    parent.setdefault(a, a)
    while parent[a] != a: a = parent[a]
    return a
for a, b, _ in edges: parent[find(a)] = find(b)
comps = {}
for a, b, _ in edges:
    for k in (a, b): comps.setdefault(find(k), set()).add(k)
comps = [sorted(v) for v in comps.values()]
print("A shock rows:", int(shock.sum()))
print("B dependency edges (z>%.0f):" % ZMIN)
for a, b, z in sorted(edges, key=lambda t: -t[2]): print(f"   {z:6.1f}  {a} x {b}")

# ---------------- C. EM mixture per component ----------------
# level-adjusted period-to-period variability ("~vol"): log std over the 4 periods minus a cubic fit on the
# row's own mean (fitted on rows not flagged so far), so it is independent of the level in normal data.
V = {}
for b in CONT:
    m = X[[f"{b}_{p}" for p in P]].values.astype(float)
    mu, l = np.nanmean(m, 1), np.log(np.nanstd(m, 1) + 1e-9)
    ok = rest & ~np.isnan(m).any(1)
    co = np.polyfit(mu[ok], l[ok], 3); v = l - np.polyval(co, mu); v[np.isnan(m).any(1)] = np.nan
    V[f"{b}~vol"] = v
FE = pd.concat([F, pd.DataFrame(V, index=F.index)], axis=1)

def em_cluster(cols, sign, mask, init=None):
    Y0 = FE.loc[mask, cols]
    Y0 = Y0.fillna(Y0.median())
    mu0, sd0 = Y0.mean(), Y0.std()
    Y = ((Y0 - mu0) / sd0).values
    bg = np.ones(len(Y))
    for j in range(Y.shape[1]):
        h, e = np.histogram(Y[:, j], bins=200); w = e[1] - e[0]
        dens = np.convolve(h, np.ones(5) / 5, "same") / (len(Y) * w)
        bg *= np.maximum(dens[np.clip(((Y[:, j] - e[0]) / w).astype(int), 0, 199)], 1e-6)
    # init: rows in the joint tail of all component features
    r = np.ones(len(Y))
    for j, s in enumerate(sign):
        r *= (s * Y[:, j] > np.quantile(s * Y[:, j], 0.9))
    r = r.astype(float) + 1e-6
    if init is not None: r = np.asarray(init, float) + 1e-6      # warm start from the previous posterior
    for _ in range(100):
        pi = r.mean(); mu = (r[:, None] * Y).sum(0) / r.sum(); d = Y - mu
        S = (r[:, None, None] * np.einsum("ij,ik->ijk", d, d)).sum(0) / r.sum() + 1e-6 * np.eye(len(cols))
        cl = multivariate_normal(mu, S, allow_singular=True).pdf(Y)
        r = pi * cl / (pi * cl + (1 - pi) * bg + 1e-300)
    post = pd.Series(0.0, index=F.index); post[mask] = r
    return post, Y0, dict(pi=float(pi), n_eff=float(r.sum()), mu=dict(zip(cols, (mu * sd0.values + mu0.values).round(3))))

res = {"id": X[ID], "shock": shock.astype(int)}
summary = {"shock": int(shock.sum()), "components": []}
for k, comp in enumerate(sorted(comps, key=len, reverse=True)):
    cols = [c for c, _ in comp]; sign = [s for _, s in comp]
    if len(set(cols)) < len(cols): continue
    post, _, info = em_cluster(cols, sign, rest)
    # signature expansion (repeated until stable): add features - 4-period means, and then the level-adjusted
    # variability of the features already in the signature - whose cluster mean differs by > 0.6 sd
    for it in range(4):
        w = post[rest].values; Fa = FE[rest]; mu_c = (Fa.mul(w, axis=0)).sum() / w.sum()
        shift = (mu_c - Fa.mean()) / Fa.std()
        pool = list(F.columns) + ([f"{c}~vol" for c in cols if f"{c}~vol" in FE] if it > 0 else [])
        extra = [(c, int(np.sign(shift[c]))) for c in pool if c not in cols and abs(shift[c]) > EXP_T]
        print(f"    expansion {it}: largest other shifts", shift.drop(cols).abs().sort_values().tail(3).round(2).to_dict(), "-> add", extra)
        if not extra: break
        if it == 0: res[f"post_C{k}_seed"] = post.round(4).values    # seed (pair-only) version, kept for comparison
        comp2 = comp + extra; cols2 = [c for c, _ in comp2]; sign2 = [s for _, s in comp2]
        post2, _, info2 = em_cluster(cols2, sign2, rest, init=post[rest].values)
        if info2["n_eff"] < 0.7 * info["n_eff"]:      # guard: expansion must not collapse the cluster
            print("    expansion rejected (n_eff %.0f -> %.0f)" % (info["n_eff"], info2["n_eff"])); break
        comp, cols, sign, post, info = comp2, cols2, sign2, post2, info2
    # C2. support estimation: injected types turned out to be independent uniform draws per period inside a
    # fixed box. Estimate the box of every per-period metric in the signature from the confident members
    # (posterior > 0.9; 0.5%/99.5% quantiles widened by 2% of the width) and flag rows whose observed periods
    # ALL lie inside (NaN periods are ignored).  Chance rate for normal rows is ~1 row (see REPORT).
    conf = rest & (post.values > 0.9); inbox = rest.copy(); bounds = {}
    for c in [c for c in cols if c in CONT]:
        m = X[[f"{c}_{p}" for p in P]].values.astype(float); v = m[conf].ravel(); v = v[~np.isnan(v)]
        lo, hi = np.quantile(v, [0.005, 0.995]); w = hi - lo; lo, hi = lo - 0.02 * w, hi + 0.02 * w
        bounds[c] = (round(float(lo), 4), round(float(hi), 4))
        inbox &= (((m >= lo) & (m <= hi)) | np.isnan(m)).all(1)
    flag = inbox if bounds else (post > TAU).values
    name = f"C{k}_" + "+".join(f"{c}{'>' if s > 0 else '<'}" for c, s in comp)
    res[f"post_C{k}"] = post.round(4).values; res[f"C{k}_post{TAU}"] = (post > TAU).values.astype(int); res[f"C{k}"] = flag.astype(int)
    info.update(name=name, box=bounds, n_flag=int(flag.sum()), n_post_tau=int((post > TAU).sum()), n_post05=int((post > 0.5).sum()), exp_tp=float(post[flag].sum()))
    print(f"    support box {bounds}: {int(flag.sum())} rows (posterior>{TAU}: {int((post > TAU).sum())}, overlap {int((flag & (post > TAU).values).sum())})")
    summary["components"].append(info)
    print(f"C{k}: {name}\n    pi={info['pi']:.4f} n_eff={info['n_eff']:.0f} flagged(box)={info['n_flag']} flagged(post>{TAU})={info['n_post_tau']} expTP={info['exp_tp']:.0f} centre={info['mu']}")

R = pd.DataFrame(res)
ccols = [c for c in R.columns if c[:1] == "C" and c[1:].isdigit()]
for i, g in enumerate(groups): R[f"G{i}"] = g["flag"].astype(int)
summary["gap_groups"] = [{"rules": g["rules"], "n": int(g["flag"].sum())} for g in groups]
R["final"] = R[["shock"] + [f"G{i}" for i in range(len(groups))] + ccols].max(axis=1)

# ---------------- D. Isolation Forest (independent check) ----------------
feat = pd.concat([((F - F.median()) / (F.quantile(.75) - F.quantile(.25))).fillna(0),
                  pd.DataFrame(np.clip(Z, -50, 50), columns=[f"z{i}" for i in range(Z.shape[1])]) / 10,
                  pd.Series(np.clip(jump, 0, 50) / 10, name="jump")], axis=1)
iso = IsolationForest(n_estimators=500, max_samples=1024, random_state=0, n_jobs=-1).fit(feat)
R["iforest"] = (-iso.score_samples(feat)).round(5)
from sklearn.metrics import roc_auc_score
print("IForest AUC vs final set: %.3f, vs shock: %.3f" % (roc_auc_score(R.final, R.iforest), roc_auc_score(R.shock, R.iforest)))
summary["iforest_auc_final"] = float(roc_auc_score(R.final, R.iforest))
R.to_csv(f"{OUT}/t5_pipeline_scores.csv", index=False)
print("final flagged:", int(R.final.sum()))
json.dump(summary, open(f"{OUT}/t5_pipeline_summary.json", "w"), indent=1, default=str)
