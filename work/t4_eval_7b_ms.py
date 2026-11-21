# 7B multi-scale / intermediate-layer heads: random OOF, hard rows, scene-group CV (sig8 / sig30), and late fusion
# (geo-mean of head probabilities) of the s10 head with the new ones. Train labels only.
import os, itertools, numpy as np, pandas as pd
from sklearn.metrics import f1_score
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values
nn1 = np.load("t4/research/groupcv/train_nn1_radio3.npy"); H = nn1 <= np.quantile(nn1, .25)
lg = lambda p: np.log(np.clip(p, 1e-9, 1)); f = lambda L, m=slice(None): f1_score(y[m], L[m].argmax(1), average="macro")
D = "t4/preds/dinov3_head"; G = "t4/research/groupcv/oof"
HEADS = {"s10": ("vit_7b_patch16_dinov3_ph", "vit_7b_patch16_dinov3_phg")}
for t in ("s05", "s15", "l50", "l75"):
    if os.path.isfile(f"{D}/vit7b_{t}_ph/d3lin_vit7b_{t}_ph_oof.npy"): HEADS[t] = (f"vit7b_{t}_ph", f"vit7b_{t}_ph")
R = {k: lg(np.load(f"{D}/{a}/d3lin_{a}_oof.npy")) for k, (a, _) in HEADS.items()}
S = {g: {k: lg(np.load(f"{G}/{b}__{g}.npy")) for k, (_, b) in HEADS.items() if os.path.isfile(f"{G}/{b}__{g}.npy")} for g in ("sig8", "sig30")}
print(f"{'heads':22s} {'OOF':>7s} {'hard':>7s} {'sig8':>7s} {'sig30':>7s}")
for r in range(1, len(HEADS) + 1):
    for ks in itertools.combinations(HEADS, r):
        if r > 1 and "s10" not in ks: continue
        L = np.mean([R[k] for k in ks], 0)
        g = [f(np.mean([S[x][k] for k in ks], 0)) if all(k in S[x] for k in ks) else float("nan") for x in ("sig8", "sig30")]
        print(f"{'+'.join(ks):22s} {f(L):7.4f} {f(L, H):7.4f} {g[0]:7.4f} {g[1]:7.4f}")
