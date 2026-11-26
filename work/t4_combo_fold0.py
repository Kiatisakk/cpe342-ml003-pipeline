# Fold-0 check of LoRA-slot combinations that exist both on fold 0 and on test (no retraining). Train prior applied to every
# part except l75la (its loss already balances it). Errors on scene fold 0.
import numpy as np, pandas as pd, itertools
from sklearn.metrics import f1_score
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values; lg = lambda p: np.log(np.clip(p, 1e-9, 1))
lpt = np.log(tr["class"].value_counts(normalize=True).reindex(C).values)
va = np.load("t4/preds/v4f_lora7b/preds/lora7b_l75_val0_val_idx.npy"); yv = y[va]
pv = lambda d, t: np.mean([lg(np.load(f"{d}/lora7b_{t}_oof_{v}.npy")) for v in ("plain", "hflip")], 0)
Q5 = "t4/preds/v4f_lora7b_q5"
V = {"soup2": pv(Q5, "soup2b_val0_refit") - lpt, "soup2v": pv(Q5, "soup2b_val0_refit2v") - lpt, "soup2avg": pv(Q5, "soup2b_val0_avghead") - lpt,
     "la": pv(Q5, "l75la_val0"), "l75": pv("t4/preds/v4f_lora7b/preds", "l75_val0") - lpt, "c30": pv("t4/preds/v4f_lora7b_q3", "c30f10r16e3_val0") - lpt}
V = {k: v - np.logaddexp.reduce(v, 1, keepdims=True) for k, v in V.items()}   # renormalise so parts mix fairly
f = lambda L: (int((L.argmax(1) != yv).sum()), round(f1_score(yv, L.argmax(1), average="macro"), 5))
R = []
for r in (1, 2, 3):
    for ks in itertools.combinations(V, r): R.append((f(np.mean([V[k] for k in ks], 0)), "+".join(ks)))
for w in (.3, .5, .7): R.append((f(w * V["soup2"] + (1 - w) * V["la"]), f"soup2 {w} / la {1-w:.1f}"))
for (e, s), n in sorted(R)[:20]: print(f"{e:3d} err  F1 {s}  {n}")
