# Compare LoRA 7B recipes on SGKF scene fold 0 and their test agreement (train labels only).
import numpy as np, pandas as pd, itertools
from sklearn.metrics import f1_score
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values; lg = lambda p: np.log(np.clip(p, 1e-9, 1))
A, Q = "t4/preds/v4f_lora7b/preds", "t4/preds/v4f_lora7b_q1/preds"; Q2 = "t4/preds/v4f_lora7b_q2/preds"
va = np.load(f"{A}/lora7b_l75_val0_val_idx.npy"); yv = y[va]
for d, t in [(Q, "c40f28r16e3"), (Q, "c24f12r16e3")]: assert (np.load(f"{d}/lora7b_{t}_val0_val_idx.npy") == va).all()
V = {k: np.mean([lg(np.load(f"{d}/lora7b_{t}_val0_oof_{v}.npy")) for v in ("plain", "hflip")], 0)
     for k, d, t in [("l75", A, "l75"), ("c40", Q, "c40f28r16e3"), ("c24", Q, "c24f12r16e3")]}
f = lambda L: round(f1_score(yv, L.argmax(1), average="macro"), 5); err = lambda L: int((L.argmax(1) != yv).sum())
for k, L in V.items(): print(f"fold0 {k}: F1 {f(L)} errors {err(L)}")
for r in (2, 3):
    for ks in itertools.combinations(V, r):
        L = np.mean([V[k] for k in ks], 0); print(f"fold0 {'+'.join(ks)}: F1 {f(L)} errors {err(L)}")
for w in (.5, .6, .7): print(f"fold0 l75 {w} + (c40+c24) {1-w:.1f}:", f(w * V["l75"] + (1 - w) * (V["c40"] + V["c24"]) / 2))
print("fold0 pairwise argmax agreement:", {f"{a}-{b}": round(float(np.mean(V[a].argmax(1) == V[b].argmax(1))), 4) for a, b in itertools.combinations(V, 2)})
T = {k: np.mean([lg(np.load(f"{d}/v4f_lora7b_{t}_test_full_{v}.npy")) for v in ("plain", "hflip")], 0)
     for k, d, t in [("l75s42", A, "l75_s42"), ("l75s7", Q, "l75_s7"), ("c40", Q, "c40f28r16e3_s42"), ("c24", Q2, "c24f12r16e3_s42")]}
print("test pairwise agreement:", {f"{a}-{b}": round(float(np.mean(T[a].argmax(1) == T[b].argmax(1))), 4) for a, b in itertools.combinations(T, 2)})
print("test conf<.6 share:", {k: round(float((np.exp(L).max(1) < .6).mean()), 4) for k, L in T.items()})
