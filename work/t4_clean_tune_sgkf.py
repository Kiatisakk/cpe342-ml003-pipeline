# Clean T4 tuning on true held-out scenes (SGKF folds 0-2, train labels only, no test data).
# FT half = scene-CV ConvNeXt-S (only FT run with scene OOF), fixed at .5 (LB-confirmed). Frozen half: 4 _ph-style heads equal +
# 7B group with share q, the 7B group itself a mix of s10 / l75 / s15 / l50 / s05. Also: per-class logit bias fitted on 2 folds,
# scored on the 3rd (nested) = a prior adjustment from our own validation split (allowed by the TA's rule).
import itertools, os, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from scipy.optimize import minimize
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values; K = len(C)
lg = lambda p: np.log(np.clip(p, 1e-9, 1)); S = "t4/research/groupcv/oof_sgkf"; FOLD = np.load("t4/research/groupcv/sgkf_sig8_fold.npy")
ft = lg(np.mean([np.load(f"t4/preds/scenecv_s/preds/v4g_cnxt_s_d3_base_oof_{v}.npy") for v in ("plain", "hflip", "up")], 0))
o4 = np.mean([lg(np.load(f"{S}/{k}.npy")) for k in ("vithp", "radio", "radio3", "dinov2g")], 0)
H7 = {"s10": "vit7b"} | {t: f"vit7b_{t}" for t in ("l75", "s15", "l50", "s05") if os.path.isfile(f"{S}/vit7b_{t}.npy")}
Z = {k: lg(np.load(f"{S}/{v}.npy")) for k, v in H7.items()}
m = ~np.isnan(Z["s10"][:, 0]) & (np.abs(ft).sum(1) > 0); fo = FOLD[m]
f = lambda L: f1_score(y[m], L[m].argmax(1), average="macro")
pf = lambda L: [f1_score(y[m][fo == k], L[m][fo == k].argmax(1), average="macro") for k in (0, 1, 2)]
def blend(q, mix, sh=.5):
    g = sum(w * Z[k] for k, w in mix.items()) / sum(mix.values())
    return (1 - sh) * ft + sh * ((1 - q) * o4 + q * g)
print("rows", m.sum(), "heads", list(H7), "| single 7B heads:", {k: round(f(v), 4) for k, v in Z.items()})
R = []
keys = list(H7); steps = [0, .25, .5, .75, 1]
for ws in itertools.product(steps, repeat=len(keys)):
    if abs(sum(ws) - 1) > 1e-9: continue
    mix = {k: w for k, w in zip(keys, ws) if w > 0}
    for q in (.5, .6, .7, .8, .85, .9):
        L = blend(q, mix); R.append((f(L), q, mix, pf(L)))
R.sort(key=lambda r: -r[0])
ref = {"v74b (s10, q .7)": blend(.7, {"s10": 1}), "v78a (l75, q .85)": blend(.85, {"l75": 1}), "s10+l75 q .7": blend(.7, {"s10": .5, "l75": .5})}
for n, L in ref.items(): print(f"{n:22s} {f(L):.4f}  folds", np.round(pf(L), 4))
print("top 12:")
for s, q, mix, p in R[:12]: print(f"  {s:.4f} q {q:.2f} {mix} folds {np.round(p, 4)}")
# nested class bias on the best blend and on v74b
def fit_bias(L, idx):
    obj = lambda b: -f1_score(y[idx], (L[idx] + np.r_[0, b]).argmax(1), average="macro")
    best = (obj(np.zeros(K - 1)), np.zeros(K - 1))
    for s in range(4):
        r = minimize(obj, np.random.default_rng(s).normal(0, .2, K - 1) * (s > 0), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 400})
        if r.fun < best[0]: best = (r.fun, r.x)
    return np.r_[0, best[1]]
mi = np.where(m)[0]
for n, L in [("v74b", ref["v74b (s10, q .7)"]), ("best grid", blend(R[0][1], R[0][2]))]:
    out = []
    for k in (0, 1, 2):
        a, b = mi[FOLD[mi] != k], mi[FOLD[mi] == k]; bb = fit_bias(L, a)
        out.append((f1_score(y[b], L[b].argmax(1), average="macro"), f1_score(y[b], (L[b] + bb).argmax(1), average="macro"), np.round(bb, 2)))
    print(f"class bias nested [{n}]:", " | ".join(f"{x:.4f}->{z:.4f} {bb}" for x, z, bb in out))
