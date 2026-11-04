"""Task 5 - discovery analyses behind REPORT.md (all numbers in the report come from this script).

Usage (from repo root):
    python t123/research/t5/t5_discovery.py [test.csv] [t5_groups.csv]

Sections
  1. Independence check: max |corr| between per-row 4-period means among rows outside the union.
  2. Group signatures: robust shift of every feature inside each known union group.
  3. Signature completeness: rows outside the union scored on each stationary group's OTHER features.
  4. Shock group: every rule row has a period-4 shock > 10 robust-z; period-1..3 shocks outside the union.
  5. Pairwise tail co-occurrence (5% tails) vs independence, outside the union (cross-metric pairs only).
  6. Triple grid scan (8x8x8 rank cells) on 4-period means, outside the union.
  7. Triple grid scans on dynamics features (period std, d12, d34, slope), outside union + new cluster.
  8. Discrete columns and flags vs every feature (contingency z), outside the union.
  9. New cluster: copula kNN density, Mahalanobis with a column-shuffle null (precision estimate).
"""
import sys, itertools, numpy as np, pandas as pd, warnings
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")
SRC = sys.argv[1] if len(sys.argv) > 1 else "cpe342-kaggle-main/05_anomaly_detection/test.csv"
GRP = sys.argv[2] if len(sys.argv) > 2 else "submissions/probes_t5/t5_groups.csv"
X = pd.read_csv(SRC); G = pd.read_csv(GRP)
u = G.iloc[:, 1:].max(axis=1).values == 1; o = ~u; r = G.rule.values == 1
P = [1, 2, 3, 4]
ALLB = sorted({c[:-2] for c in X.columns if c[-2:] in ("_1", "_2", "_3", "_4")})
DISC = ["device_count", "refund_count", "guild_changes"]
FLAGS = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
CONT = [b for b in ALLB if b not in ["login_lat", "login_lon"] + DISC]
M = pd.DataFrame({b: X[[f"{b}_{p}" for p in P]].mean(axis=1) for b in ALLB if b not in ("login_lat", "login_lon")})
M["account_age_days"] = X.account_age_days; M["behavior_stability"] = X.behavior_stability
N = M[o]; Zr = (M - N.median()) / ((N.quantile(.75) - N.quantile(.25)) / 1.349)
H = lambda s: print("\n" + "=" * 8, s, "=" * 8)

H("1. independence of per-row means (outside union)")
C = M[o].corr().values; np.fill_diagonal(C, 0) if C.flags.writeable else None
C = np.where(np.eye(len(C), dtype=bool), 0, C)
print("max |corr| between different features: %.3f" % np.abs(C).max())

H("2. group signatures (median robust-z vs outside rows, |z|>0.4, continuous features)")
for g in ["rule", "bot", "lowact", "lowcomm", "lowstab"]:
    d = Zr[G[g] == 1][[c for c in Zr.columns if c not in DISC]].median()
    print(g, int((G[g] == 1).sum()), d[d.abs() > 0.4].sort_values().round(2).to_dict())

H("3. signature completeness: outside rows scoring like the group on its other features")
sig = {"bot": {"ip_hash_entropy": -1, "login_time_variance": -1, "session_duration_std": -1, "kda_std": -1, "performance_stability": 1, "play_pattern_consistency": 1},
       "lowact": {"activity_level": -1, "purchase_amount_mean": 1, "skill_rating": 1},
       "lowcomm": {"communication_frequency": -1, "friend_adds": 1},
       "lowstab": {"behavior_stability": -1, "login_count": -1, "purchase_frequency": 1}}
for g, s in sig.items():
    for k in s:
        rest = [j for j in s if j != k]
        sc = sum(Zr[j] * s[j] for j in rest) / len(rest)
        print(f"{g:8s} leave-out {k:24s} group median {np.nanmedian(sc[G[g] == 1]):5.2f} | outside 99.9% {np.nanquantile(sc[o], .999):5.2f}")

H("4. shock group: robust-z of period-q deviation from the mean of the other periods")
D = {}
for b in [b for b in ALLB if b not in ("login_lat", "login_lon")]:
    m = X[[f"{b}_{p}" for p in P]].values.astype(float)
    for q in P:
        d = m[:, q - 1] - np.nanmean(m[:, [i for i in range(4) if i != q - 1]], 1)
        nd = d[o]; med = np.nanmedian(nd); mad = 1.4826 * np.nanmedian(np.abs(nd - med))
        D[f"{b}@{q}"] = (d - med) / mad
D = pd.DataFrame(D)
A4 = D[[c for c in D.columns if c.endswith("@4") and not c.startswith("location")]].abs()
n10 = (A4 > 10).sum(axis=1)
print("rule rows by #metrics with |z|>10 at p4:", pd.Series(n10[r]).value_counts().sort_index().to_dict())
print("outside rows with any |z|>10 at p4:", int((n10[o] > 0).sum()), "->", A4[o & (n10 > 0)].idxmax(axis=1).value_counts().to_dict())
for q in [1, 2, 3]:
    fq = [c for c in D.columns if c.endswith(f"@{q}") and not c.startswith("location")]
    print(f"period {q}: outside rows with any |z|>10: {int(((D.loc[o, fq].abs() > 10).sum(axis=1) > 0).sum())}")

H("5. pairwise 5%-tail co-occurrence vs independence (outside union, different metrics)")
SD = pd.DataFrame({b + "~sd": X[[f"{b}_{p}" for p in P]].std(axis=1) for b in CONT})
FF = pd.concat([Zr[[c for c in Zr.columns if c not in DISC]], SD], axis=1)[o]
T = {}
for c in FF.columns:
    T[c + "<"] = FF[c] < FF[c].quantile(.05); T[c + ">"] = FF[c] > FF[c].quantile(.95)
for f in FLAGS: T[f] = X.loc[o, f] == 1
T = pd.DataFrame(T).astype(float); cnt = (T.T @ T).values; p = T.mean().values; E = np.outer(p, p) * len(T)
Zp = (cnt - E) / np.sqrt(E + 1e-9); nm = list(T.columns)
base = lambda s: s.rstrip("<>").split("~")[0]
res = sorted([(Zp[i, j], nm[i], nm[j], int(cnt[i, j]), E[i, j]) for i in range(len(nm)) for j in range(i + 1, len(nm)) if base(nm[i]) != base(nm[j])], reverse=True)
for x in res[:6]: print(f"z={x[0]:5.1f}  {x[1]} x {x[2]}  obs {x[3]} exp {x[4]:.1f}")

def triple_scan(F, B):
    R = F.rank(pct=True, method="first").values; ok = ~F.isna().values
    Q = np.minimum((np.nan_to_num(R) * B).astype(int), B - 1); n = F.shape[1]; out = []
    for i, j, k in itertools.combinations(range(n), 3):
        v = ok[:, i] & ok[:, j] & ok[:, k]
        c = np.bincount(Q[v, i] * B * B + Q[v, j] * B + Q[v, k], minlength=B ** 3); e = v.sum() / B ** 3
        out.append(((c.max() - e) / np.sqrt(e), F.columns[i], F.columns[j], F.columns[k], int(c.max()), round(e, 1)))
    return sorted(out, key=lambda t: -t[0])

H("6. triple grid scan on 4-period means (outside union)")
F6 = M[[c for c in M.columns if c not in DISC]][o]
t6 = triple_scan(F6, 8)
for x in t6[:4]: print(f"z={x[0]:5.1f}", x[1:])
print("median z over %d triples: %.2f" % (len(t6), np.median([x[0] for x in t6])))

# new cluster (needed for section 7 exclusion)
cols = ["accuracy_std", "kda_std", "performance_stability"]
Fo = M.loc[o, cols].dropna(); Rk = Fo.rank(pct=True).values
d, _ = NearestNeighbors(n_neighbors=31).fit(Rk).kneighbors(Rk)
dens = pd.Series(30 / (len(Rk) * 4 / 3 * np.pi * d[:, -1] ** 3), index=Fo.index)

H("7. triple grid scans on dynamics features (outside union and outside new cluster)")
o7 = o.copy(); o7[dens.index[dens > 1.5]] = False
for kind in ["sd", "d12", "d34", "slope"]:
    Fk = {}
    for b in CONT:
        m = X[[f"{b}_{p}" for p in P]].values
        Fk[b] = {"sd": np.nanstd(m, 1) / (np.abs(np.nanmean(m, 1)) + 1e-3), "d12": m[:, 1] - m[:, 0],
                 "d34": m[:, 3] - m[:, 2], "slope": m[:, 2] + m[:, 3] - m[:, 0] - m[:, 1]}[kind]
    t7 = triple_scan(pd.DataFrame(Fk)[o7], 6)
    print(f"{kind:5s} top z={t7[0][0]:4.1f} {t7[0][1:4]}  median z={np.median([x[0] for x in t7]):.2f}")

H("8. discrete columns + flags vs every feature decile / each other (outside union)")
Dd = pd.DataFrame({f"{b}_{p}": X.loc[o, f"{b}_{p}"] for b in DISC for p in P});
for f in FLAGS: Dd[f] = X.loc[o, f]
Qd = M.loc[o, [c for c in M.columns if c not in DISC]].apply(lambda c: pd.qcut(c, 10, labels=False, duplicates="drop"))
best = []
for a in Dd.columns:
    for b in list(Qd.columns) + list(Dd.columns):
        if b == a or b.rsplit("_", 1)[0] == a.rsplit("_", 1)[0] or a.startswith(b): continue
        t = pd.crosstab(Dd[a], Qd[b] if b in Qd else Dd[b]); e = np.outer(t.sum(axis=1), t.sum(axis=0)) / t.values.sum()
        best.append((np.abs((t.values - e) / np.sqrt(e + 1e-9)).max(), a, b))
best.sort(reverse=True)
print("top 3:", [(round(x[0], 1), x[1], x[2]) for x in best[:3]])
for b in DISC:
    Rr = X[[f"{b}_{p}" for p in P]].values
    d14 = Rr[:, 3] - Rr[:, 0]
    print(f"{b}: p4-p1 values outside union {sorted(set(d14[o & ~np.isnan(d14)]))}, in union {sorted(set(d14[u & ~np.isnan(d14)]))}")

H("9. new cluster accuracy_std + kda_std + performance_stability")
print("copula density histogram (log2):", np.histogram(np.log2(dens), bins=np.arange(-3, 6.5, 0.5))[0].tolist())
for t in [2, 4]: print(f"density>{t}: {int((dens > t).sum())} rows, centre {M.loc[dens.index[dens > t], cols].mean().round(3).to_dict()}")
Zs = (M[cols] - Fo.mean()) / Fo.std(); seed = dens.index[dens > 4]
mu = Zs.loc[seed].mean().values; Ci = np.linalg.inv(np.cov(Zs.loc[seed].values.T))
md = lambda Z: np.sqrt(np.einsum("ij,jk,ik->i", Z - mu, Ci, Z - mu))
Dm = pd.Series(md(Zs.values), index=M.index)
rng = np.random.default_rng(0)
null = np.concatenate([md(np.stack([rng.permutation(Zs.loc[o, c].dropna().values)[:o.sum() - 200] for c in cols], 1)) for _ in range(20)])
scale = (o.sum()) / (o.sum() - 200)
print("Mahalanobis radius | outside rows | shuffle-null expected | excess | in union")
for t in [1.5, 2, 2.5, 3, 3.5, 4, 5]:
    n = int((Dm[o] < t).sum()); e = (null < t).sum() / 20 * scale
    print(f"  <{t:3}: {n:4d}  {e:6.1f}  {n - e:6.1f}  {int((Dm[u] < t).sum())}")
