# Post-hoc variants on the v52 model set (FT: v3, ConvNeXt-B, Colab L, EVA02-B; frozen: v37 4-view heads with v37 weights),
# scored on OOF (rows covered by every run) and on "hard" rows (bottom 25% train->train NN cosine). EM is applied to the
# test output only, so it cannot be scored here; these numbers compare the pre-EM blends.
#   geo     : weighted geometric mean (= v52)
#   arith   : weighted arithmetic mean of probabilities
#   calib   : per-model temperature (fitted by NLL on OOF) then geo
#   stack   : multinomial logistic regression on calibrated log-probs (nested 5-fold on OOF rows)
#   +mlp    : concat-feature MLP head added to the frozen group (if finished)
import os, sys, numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values; K = len(C)
V = ["plain", "hflip", "up"]; lg = lambda p: np.log(np.clip(p, 1e-9, 1))
def O(d, t, vs): return np.mean([np.load(f"{d}/{t}_oof_{v}.npy") for v in vs], 0) if vs else np.load(f"{d}/{t}_oof.npy")
M = [("v3", "t4/preds/v3_5f/preds", "v3_5f", [], .125), ("cnxtB", "t4/preds/v4_cnxt_b_d3/preds", "v4_cnxt_b_d3", V, .125),
     ("cnxtLc", "t4/preds/v4_cnxt_l_d3_colab/preds", "v4_cnxt_l_d3c", V, .125), ("eva02b", "t4/preds/v4_eva02_b/preds", "v4_eva02_b", V, .125),
     ("vithp", "t4/preds/dinov3_head/vithp", "d3lin_vithp", [], .20), ("radio", "t4/preds/dinov3_head/radio", "d3lin_radio", [], .05),
     ("radio3", "t4/preds/dinov3_head/radio3", "d3lin_radio3", [], .10), ("dinov2g", "t4/preds/dinov3_head/dinov2g", "d3lin_dinov2g", [], .15)]
P = {k: O(d, t, vs) for k, d, t, vs, _ in M}; W = {k: w for k, *_, w in M}
mlp = "t4/preds/dinov3_head/concat4_mlp/d3lin_concat4_mlp_oof.npy"
if os.path.exists(mlp): P["mlp"] = np.load(mlp)
m = np.all([~np.isnan(p[:, 0]) for p in P.values()], 0)
nn1 = np.load("t4/research/groupcv/train_nn1_radio3.npy"); hard = nn1 <= np.quantile(nn1, .25)
idx = np.where(m)[0]; yy = y[idx]; hh = hard[idx]
L = {k: lg(v[idx]) for k, v in P.items()}
def sc(z): return f1_score(yy, z.argmax(1), average="macro"), f1_score(yy[hh], z[hh].argmax(1), average="macro")
def temp(z):
    zt = torch.tensor(z); yt = torch.tensor(yy); lt = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([lt], lr=.1, max_iter=200)
    def cl(): opt.zero_grad(); l = torch.nn.functional.cross_entropy(zt / lt.exp(), yt); l.backward(); return l
    opt.step(cl); return float(lt.detach().exp())
T = {k: temp(L[k]) for k in L}
print("rows", len(idx), "hard", hh.sum(), "| temperatures", {k: round(v, 2) for k, v in T.items()})
res = {}
base = [k for k, *_ in M]
res["geo (v52)"] = sc(sum(W[k] * L[k] for k in base))
res["arith"] = sc(np.log(sum(W[k] * np.exp(L[k]) for k in base)))
res["calib geo"] = sc(sum(W[k] * L[k] / T[k] for k in base))
fr = ["vithp", "radio", "radio3", "dinov2g"]
for share in (.4, .6):                                  # FT/frozen balance check (LB chose .5)
    w2 = {k: (1 - share) / 4 for k in base[:4]}; w2.update({k: W[k] / .5 * share for k in fr})
    res[f"geo frozen {share:.1f}"] = sc(sum(w2[k] * L[k] for k in base))
if "mlp" in L:
    w3 = dict(W); [w3.__setitem__(k, W[k] * .8) for k in fr]; w3["mlp"] = .1
    res["geo + mlp (.1)"] = sc(sum(w3[k] * L[k] for k in w3))
X = np.concatenate([L[k] / T[k] for k in base], 1); st = np.zeros((len(idx), K))
for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, yy):
    st[b] = LogisticRegression(C=.1, max_iter=3000).fit(X[a], yy[a]).predict_proba(X[b])
res["stack (nested)"] = sc(np.log(st + 1e-9))
print(f"\n{'variant':18s} {'OOF':>8s} {'hard':>8s}")
for k, (a, b) in res.items(): print(f"{k:18s} {a:.5f} {b:.5f}")
