# Task 3 — Neural net + ZILN / Mixture-of-Lognormals head (NMAE)
# Backbone: numeric PLR (periodic) embeddings + residual MLP; head: logit p | K x (pi, mu, sigma) on log1p(y)
# Loss = BCE(pay?) + NLL(mixture lognormal | payers)   (K=1 == ZILN, Wang et al. 2019; K>1 ~ OptDist-style mixture)
# Prediction = MAE-optimal: p<=0.5 -> 0 else mixture quantile at q = 1 - 1/(2p)   (+ OOF-tuned p-shift / kappa)
# Same preprocessing as GBM v2-v4 (exact identities + segment imputation), same 5 folds -> OOF blends with GBM OOF if attached.
# NO SUBMIT - writes /kaggle/working/preds only
import os, time, json, math, warnings, numpy as np, pandas as pd, torch, torch.nn as nn, xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
warnings.filterwarnings("ignore")
SEED, NF = 42, 5
KS = [1, 3]                    # mixture components to compare (1 = plain ZILN)
SEEDS = [0, 1]                 # NN seeds averaged per fold
EPOCHS, PATIENCE, BS, LR, WD = 80, 10, 1024, 2e-3, 1e-5
WIDTH, DEPTH, DROP, NFREQ, DEMB = 512, 3, 0.15, 16, 16
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE: KS, SEEDS, EPOCHS, PATIENCE, WIDTH = [1, 3], [0], 3, 2, 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.environ.get("DATA_ROOT", "/kaggle/input"); OUT = os.environ.get("OUT_DIR", "/kaggle/working/preds")
print("torch", torch.__version__, "| device", DEV, flush=True)

def walk_find(name, must=None):
    for d, _, fs in os.walk(ROOT, followlinks=True):
        if name in fs and (must is None or must in d): return os.path.join(d, name)
for _ in range(30):
    TRAIN, TEST = walk_find("train.csv", "03_regression"), walk_find("test.csv", "03_regression")
    if TRAIN and TEST: break
    time.sleep(10)
SAMPLE = walk_find("sample_submission.csv")
assert TRAIN and TEST, "data not found - Add Input -> competition data"
tr, te = pd.read_csv(TRAIN), pd.read_csv(TEST)
if SMOKE: tr = tr.sample(8000, random_state=0).reset_index(drop=True)
TARGET = "spending_30d"; y = tr[TARGET].values.astype(float); pos = y > 0
ids = pd.read_csv(SAMPLE)["id"].values if SAMPLE else np.array([f"ANS{i+1:05d}" for i in range(len(te))])
def nmae(yt, yp): return 1 / (1 + np.abs(yt - yp).sum() / yt.sum())
print(tr.shape, te.shape, flush=True)

# ---------- 1. same preprocessing as GBM v2-v4 ----------
def exact_fill(d):
    d = d.copy()
    for _ in range(3):
        h, tx, av = d["historical_spending"], d["total_transactions"], d["avg_transaction_value"]
        po, dr = d["purchases_on_discount"], d["discount_rate_used"]
        for col, m, val in [("historical_spending", h.isna() & tx.notna() & av.notna(), tx * av),
                            ("total_transactions", tx.isna() & h.notna() & av.notna() & (av > 0), h / av),
                            ("avg_transaction_value", av.isna() & h.notna() & tx.notna() & (tx > 0), h / tx),
                            ("purchases_on_discount", po.isna() & tx.notna() & dr.notna(), tx * dr),
                            ("total_transactions", tx.isna() & po.notna() & dr.notna() & (dr > 0), po / dr),
                            ("discount_rate_used", dr.isna() & po.notna() & tx.notna() & (tx > 0), po / tx)]:
            m = m & d[col].isna(); d.loc[m, col] = val[m]
    return d
tr, te = exact_fill(tr), exact_fill(te)
def add_feats(d):
    d = d.copy(); e = 1.0
    d["hist_per_day"] = d["historical_spending"] / (d["account_age_days"] + e)
    d["tx_per_day"] = d["total_transactions"] / (d["account_age_days"] + e)
    d["exp_monthly_spend"] = d["avg_transaction_value"] * d["purchase_frequency"]
    d["prev_vs_hist"] = d["prev_month_spending"] / (d["historical_spending"] + e)
    d["prev_x_freq"] = d["prev_month_spending"] * d["purchase_frequency"]
    d["recency_ratio"] = d["days_since_last_purchase"] / (d["days_since_last_login"] + e)
    d["play_per_session"] = d["total_playtime_hours"] / (d["sessions_per_week"] + e)
    d["n_missing"] = d.isna().sum(axis=1)
    return d
X, Xt = add_feats(tr.drop(columns=[TARGET])), add_feats(te)
FEATS = [c for c in Xt.columns if c != "player_id"]
X, Xt = X[FEATS].copy(), Xt[FEATS].copy()
SF = [c for c in FEATS if c != "segment"]; known = X["segment"].notna().values
sc = xgb.XGBClassifier(objective="multi:softprob", n_estimators=60 if SMOKE else 4000, learning_rate=0.02, early_stopping_rounds=100,
                       max_depth=6, subsample=0.8, colsample_bytree=0.7, tree_method="hist", device=DEV, random_state=SEED)
i_a, i_b = train_test_split(np.where(known)[0], test_size=0.1, random_state=SEED, stratify=X["segment"].values[known])
sc.fit(X.iloc[i_a][SF], X["segment"].values[i_a].astype(int), eval_set=[(X.iloc[i_b][SF], X["segment"].values[i_b].astype(int))], verbose=False)
for D in (X, Xt):
    miss = D["segment"].isna().values; D["seg_imputed"] = miss.astype(int)
    if miss.any(): D.loc[miss, "segment"] = sc.predict_proba(D.loc[miss, SF]).argmax(1)

# ---------- 2. NN input: low-cardinality -> one-hot, others -> quantile-normal + NaN indicator ----------
CAT = [c for c in X.columns if X[c].nunique(dropna=True) <= 12]
NUM = [c for c in X.columns if c not in CAT]
def onehot(A, B):
    oa, ob = [], []
    for c in CAT:
        lv = sorted(pd.concat([A[c], B[c]]).dropna().unique())
        for v in lv: oa.append((A[c] == v).values); ob.append((B[c] == v).values)
        oa.append(A[c].isna().values); ob.append(B[c].isna().values)
    return np.stack(oa, 1).astype(np.float32), np.stack(ob, 1).astype(np.float32)
Ca, Ct = onehot(X, Xt)
qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200000, random_state=SEED).fit(X[NUM])   # features only, train only
Na, Nt = qt.transform(X[NUM]).astype(np.float32), qt.transform(Xt[NUM]).astype(np.float32)
Ma, Mt = np.isnan(Na).astype(np.float32), np.isnan(Nt).astype(np.float32)
Na, Nt = np.nan_to_num(Na), np.nan_to_num(Nt)
keepM = Ma.any(0) | Mt.any(0); Ma, Mt = Ma[:, keepM], Mt[:, keepM]
print(f"NN input: {len(NUM)} numeric (PLR) + {Ca.shape[1]} one-hot + {Ma.shape[1]} NaN-flags", flush=True)

# ---------- 3. model ----------
class PLR(nn.Module):                      # periodic embeddings for numerical features (Gorishniy et al. 2022)
    def __init__(self, nf, k=NFREQ, d=DEMB, sigma=0.1):
        super().__init__(); self.c = nn.Parameter(torch.randn(nf, k) * sigma); self.lin = nn.Linear(2 * k, d)
    def forward(self, x):                  # x: (B, nf)
        v = 2 * math.pi * x[..., None] * self.c
        return torch.relu(self.lin(torch.cat([torch.sin(v), torch.cos(v)], -1))).flatten(1)
class Block(nn.Module):
    def __init__(self, w, p):
        super().__init__(); self.f = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.GELU(), nn.Dropout(p), nn.Linear(w, w))
    def forward(self, x): return x + self.f(x)
class Net(nn.Module):
    def __init__(self, nnum, nother, K):
        super().__init__(); self.K = K; self.plr = PLR(nnum)
        self.inp = nn.Linear(nnum * DEMB + nother, WIDTH)
        self.body = nn.Sequential(*[Block(WIDTH, DROP) for _ in range(DEPTH)], nn.LayerNorm(WIDTH))
        self.head = nn.Linear(WIDTH, 1 + 3 * K)
    def forward(self, xn, xo):
        h = self.body(self.inp(torch.cat([self.plr(xn), xo], 1)))
        o = self.head(h); K = self.K
        logit, wl, mu, s = o[:, 0], o[:, 1:1+K], o[:, 1+K:1+2*K], o[:, 1+2*K:]
        return logit, torch.log_softmax(wl, 1), mu + 8.0, nn.functional.softplus(s) + 0.05   # mu offset ~ mean log1p(y | pay)
def ziln_loss(logit, lw, mu, sig, pay, ly):
    bce = nn.functional.binary_cross_entropy_with_logits(logit, pay)
    m = pay > 0
    if m.sum() == 0: return bce
    z = (ly[m, None] - mu[m]) / sig[m]
    comp = lw[m] - torch.log(sig[m]) - 0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)
    return bce + (-torch.logsumexp(comp, 1)).mean() * m.float().mean()

def predict_params(net, xn, xo):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(xn), 8192):
            r = net(torch.from_numpy(xn[i:i+8192]).to(DEV), torch.from_numpy(xo[i:i+8192]).to(DEV))
            out.append([t.float().cpu().numpy() for t in (torch.sigmoid(r[0]), r[1].exp(), r[2], r[3])])
    return [np.concatenate([o[j] for o in out]) for j in range(4)]

def train_one(K, seed, idx_tr, idx_es, Xn, Xo):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(Xn.shape[1], Xo.shape[1], K).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * math.ceil(len(idx_tr) / BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
    T = lambda a, i: torch.from_numpy(a[i]).to(DEV)
    payf, lyf = pos.astype(np.float32), np.log1p(y).astype(np.float32)
    best, bstate, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(idx_tr)
        for i in range(0, len(perm), BS):
            bi = perm[i:i+BS]
            loss = ziln_loss(*net(T(Xn, bi), T(Xo, bi)), T(payf, bi), T(lyf, bi))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
        net.eval()
        with torch.no_grad(): vl = float(ziln_loss(*net(T(Xn, idx_es), T(Xo, idx_es)), T(payf, idx_es), T(lyf, idx_es)))
        if vl < best - 1e-5: best, bstate, bad = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    net.load_state_dict(bstate); return net, best, ep + 1

# ---------- 4. mixture quantile -> MAE-optimal prediction ----------
from scipy.special import ndtr
def mix_quantile(w, mu, sig, q, iters=50):
    lo = (mu - 6 * sig).min(1); hi = (mu + 6 * sig).max(1)
    for _ in range(iters):
        mid = (lo + hi) / 2
        cdf = (w * ndtr((mid[:, None] - mu) / sig)).sum(1)
        lo, hi = np.where(cdf < q, mid, lo), np.where(cdf < q, hi, mid)
    return (lo + hi) / 2
def pred_mix(P, shift=0.0, kap=1.0):
    p, w, mu, sig = P; pp = np.clip(p + shift, 1e-6, 1)
    q = np.clip(1 - 1 / (2 * pp), 1e-4, 0.5)
    v = kap * np.expm1(mix_quantile(w, mu, sig, q))
    return np.where(pp > 0.5, np.clip(v, 0, 300000), 0.0)

# ---------- 5. CV (same folds as GBM v1-v4) ----------
Xn_tr, Xo_tr = Na, np.concatenate([Ca, Ma], 1); Xn_te, Xo_te = Nt, np.concatenate([Ct, Mt], 1)
strat = np.where(pos, 1 + pd.qcut(pd.Series(y).where(pos).rank(method="first"), 4, labels=False).fillna(-1).values, 0).astype(int)
folds = list(StratifiedKFold(NF, shuffle=True, random_state=SEED).split(X, strat))
RES, OOF, TEST = {}, {}, {}
for K in KS:
    oof = [np.zeros(len(y)), np.zeros((len(y), K)), np.zeros((len(y), K)), np.zeros((len(y), K))]
    tst = [np.zeros(len(Xt)), np.zeros((len(Xt), K)), np.zeros((len(Xt), K)), np.zeros((len(Xt), K))]
    t0 = time.time()
    for k, (a, b) in enumerate(folds):
        i_tr, i_es = train_test_split(a, test_size=0.1, random_state=SEED, stratify=strat[a])
        for sd in SEEDS:
            net, vl, ne = train_one(K, sd, i_tr, i_es, Xn_tr, Xo_tr)
            Pb = predict_params(net, Xn_tr[b], Xo_tr[b]); Pt = predict_params(net, Xn_te, Xo_te)
            # average p across seeds; keep mixture components of all seeds (concat -> mixture of seeds)
            oof[0][b] += Pb[0] / len(SEEDS); tst[0] += Pt[0] / (len(SEEDS) * NF)
            if sd == SEEDS[0]: Bw, Bm, Bs, Tw, Tm, Ts = [Pb[1] / len(SEEDS)], [Pb[2]], [Pb[3]], [Pt[1] / len(SEEDS)], [Pt[2]], [Pt[3]]
            else: Bw.append(Pb[1] / len(SEEDS)); Bm.append(Pb[2]); Bs.append(Pb[3]); Tw.append(Pt[1] / len(SEEDS)); Tm.append(Pt[2]); Ts.append(Pt[3])
            print(f"K={K} fold {k} seed {sd}: val loss {vl:.4f} | epochs {ne} | {time.time()-t0:.0f}s", flush=True)
        if k == 0: oof[1:] = [np.zeros((len(y), K * len(SEEDS))) for _ in range(3)]; TST_parts = []
        oof[1][b], oof[2][b], oof[3][b] = np.concatenate(Bw, 1), np.concatenate(Bm, 1), np.concatenate(Bs, 1)
        TST_parts.append((np.concatenate(Tw, 1), np.concatenate(Tm, 1), np.concatenate(Ts, 1)))
    # test: mixture over folds (each fold weight 1/NF)
    tst = [tst[0], np.concatenate([p[0] / NF for p in TST_parts], 1), np.concatenate([p[1] for p in TST_parts], 1), np.concatenate([p[2] for p in TST_parts], 1)]
    RES[f"NN_K{K}_theory"] = nmae(y, pred_mix(oof))
    SH, KAP = np.arange(-0.1, 0.101, 0.025), np.arange(0.85, 1.151, 0.025)
    g = max(((sh, ka) for sh in SH for ka in KAP), key=lambda t: nmae(y, pred_mix(oof, *t)))
    OOF[f"NN_K{K}"] = pred_mix(oof, *g); TEST[f"NN_K{K}"] = pred_mix(tst, *g)
    RES[f"NN_K{K}_tuned"] = nmae(y, OOF[f"NN_K{K}"])
    print(f"K={K}: theory {RES[f'NN_K{K}_theory']:.4f} | tuned {RES[f'NN_K{K}_tuned']:.4f} (shift, kappa)={np.round(g, 3)} | p logloss "
          f"{-np.mean(pos*np.log(np.clip(oof[0],1e-6,1))+(1-pos)*np.log(np.clip(1-oof[0],1e-6,1))):.4f}", flush=True)

# ---------- 6. blend with GBM OOF if a GBM output is attached (same row order & folds) ----------
gbm = None
for nm in ["t3_v4_oof.csv", "t3_v3_oof.csv", "t3_v2_oof.csv"]:
    f = walk_find(nm)
    if f: gbm = (nm, f); break
M, TM = dict(OOF), dict(TEST)
if gbm and not SMOKE:
    g_oof = pd.read_csv(gbm[1])
    if len(g_oof) == len(y) and np.allclose(g_oof["y"].values, y):
        tag = gbm[0].split("_")[1]                     # v3 / v4
        for c in [c for c in g_oof.columns if c not in ("y", "p", "px", "pl", "mu", "sig", "muw")]:
            tf = walk_find(f"t3_{tag}_{c}.csv") or walk_find(f"t3_{tag}_{c}_tuned.csv")
            if tf: M[f"GBM_{c}"] = g_oof[c].values; TM[f"GBM_{c}"] = pd.read_csv(tf)["pred"].values
        print("attached GBM:", gbm[0], "->", [k for k in M if k.startswith("GBM_")])
    else: print("GBM OOF row order mismatch - skip blend")
else: print("no GBM OOF attached - NN only (attach the v3/v4 notebook output to blend)")
for k, v in M.items(): RES.setdefault(k, nmae(y, v))
cur, cnt, picks, best_s, best_w = np.zeros(len(y)), 0, [], -1, None
for _ in range(25):
    kb = max(M, key=lambda k: nmae(y, (cur * cnt + M[k]) / (cnt + 1)))
    cur = (cur * cnt + M[kb]) / (cnt + 1); cnt += 1; picks.append(kb)
    if nmae(y, cur) > best_s: best_s, best_w = nmae(y, cur), {k: picks.count(k) / len(picks) for k in set(picks)}
RES["blend"] = best_s
print("blend weights:", {k: round(v, 2) for k, v in best_w.items()})
print("reference GBM: v2 0.7916 | v3 0.7917")
for k, v in RES.items(): print(f"{k:22s} {v:.4f}  (err ratio {1/v-1:.4f})")

# ---------- 7. save ----------
os.makedirs(OUT, exist_ok=True)
pred = np.round(sum(w * TM[k] for k, w in best_w.items()), 2)
pd.DataFrame({"id": ids, "pred": pred}).to_csv(f"{OUT}/t3_nn_blend.csv", index=False)
for k in OOF: pd.DataFrame({"id": ids, "pred": np.round(TEST[k], 2)}).to_csv(f"{OUT}/t3_{k}.csv", index=False)
pd.DataFrame({"y": y, **OOF}).to_csv(f"{OUT}/t3_nn_oof.csv", index=False)
json.dump({"oof_nmae": {k: round(float(v), 5) for k, v in RES.items()}, "blend_w": best_w}, open(f"{OUT}/t3_nn_summary.json", "w"), indent=1)
print("USE: blend | OOF NMAE", round(best_s, 4), "| test zero%", round((pred == 0).mean(), 3), "mean", round(pred.mean(), 1))
print("wrote", OUT, "- nothing submitted")
