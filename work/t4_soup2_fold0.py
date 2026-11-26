# Fold-0 table for the soup round 2 (refit-only controls, 2-view head, logit-adjusted soups), raw and x train_prior^-1.
import numpy as np, pandas as pd, os, itertools
from sklearn.metrics import f1_score
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values; lg = lambda p: np.log(np.clip(p, 1e-9, 1))
pt = tr["class"].value_counts(normalize=True).reindex(C).values
D = "t4/preds/v4f_lora7b_q5"; va = np.load("t4/preds/v4f_lora7b/preds/lora7b_l75_val0_val_idx.npy"); yv = y[va]
f = lambda L: (round(f1_score(yv, L.argmax(1), average="macro"), 5), int((L.argmax(1) != yv).sum()))
pv = lambda t: np.mean([lg(np.load(f"{D}/lora7b_{t}_oof_{v}.npy")) for v in ("plain", "hflip")], 0)
rows = []
for s in ("single_l75", "single_c30f10", "soup2b", "soup2la", "soup3la"):
    for h in ("avghead", "refit", "refit2v"):
        t = f"{s}_val0_{h}"
        if not os.path.exists(f"{D}/lora7b_{t}_oof_plain.npy"): continue
        assert (np.load(f"{D}/lora7b_{s}_val0_val_idx.npy") == va).all()
        L = pv(t); rows.append((s, h, *f(L), *f(L - np.log(pt))))
L = pv("l75la_val0"); rows.append(("l75la", "own (balanced)", *f(L), *f(L - np.log(pt))))
print(pd.DataFrame(rows, columns=["model", "head", "F1 raw", "err raw", "F1 prior", "err prior"]).to_string(index=False))
P = lambda d, t: np.mean([lg(np.load(f"{d}/v4f_lora7b_{t}_test_full_{v}.npy")) for v in ("plain", "hflip")], 0)
T = {"soup2": P("t4/preds/v4f_lora7b_q4", "soup2_s42"), "soup2hv2": P(D, "soup2hv2_s42"), "l75la": P(D, "l75la_s42"), "l75": P("t4/preds/v4f_lora7b/preds", "l75_s42")}
print("test agree", {f"{a}-{b}": round(float((T[a].argmax(1) == T[b].argmax(1)).mean()), 4) for a, b in itertools.combinations(T, 2)})
