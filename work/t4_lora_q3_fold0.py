# Fold-0 check of the c30f10 LoRA run and multi-scale TTA of the l75 LoRA (train labels only), + test agreement.
import numpy as np, pandas as pd, itertools
from sklearn.metrics import f1_score
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values; lg = lambda p: np.log(np.clip(p, 1e-9, 1))
A, D = "t4/preds/v4f_lora7b/preds", "t4/preds/v4f_lora7b_q3"
va = np.load(f"{A}/lora7b_l75_val0_val_idx.npy"); yv = y[va]
assert (np.load(f"{D}/lora7b_c30f10r16e3_val0_val_idx.npy") == va).all()
f = lambda L: round(f1_score(yv, L.argmax(1), average="macro"), 5); err = lambda L: int((L.argmax(1) != yv).sum())
pv = lambda d, t: np.mean([lg(np.load(f"{d}/lora7b_{t}_val0_oof_{v}.npy")) for v in ("plain", "hflip")], 0)
V = {"l75": pv(A, "l75"), "c30f10": pv(D, "c30f10r16e3")}
S = ["224x416", "192x352", "256x480", "288x528"]
for s in S: V[f"tta{s}"] = lg(np.load(f"{D}/lora7b_l75_val0_tta_{s}.npy"))
for k, L in V.items(): print(f"{k:12s} F1 {f(L)} err {err(L)}")
for ks in [S, S[:1] + S[2:], S[:1] + S[2:3], S[2:], S[:1] + S[3:]]:
    L = np.mean([V[f"tta{s}"] for s in ks], 0); print("TTA", "+".join(ks), f(L), err(L))
L = (V["l75"] + V["c30f10"]) / 2; print("l75+c30f10", f(L), err(L))
print("agree l75-c30f10 fold0", round(float((V["l75"].argmax(1) == V["c30f10"].argmax(1)).mean()), 4))
pt = lambda t, sfx="": np.mean([lg(np.load(f"{D}/v4f_lora7b_{t}_test_full_{v}{sfx}.npy")) for v in ("plain", "hflip")], 0)
T = {"l75": pt("l75_s42"), "c30f10": pt("c30f10r16e3_s42")}
for s in S[1:]: T[s] = pt("l75_s42", "_" + s)
print("test agree", {f"{a}-{b}": round(float(np.mean(T[a].argmax(1) == T[b].argmax(1))), 4) for a, b in itertools.combinations(T, 2)})
old = np.mean([lg(np.load(f"{A}/v4f_lora7b_l75_s42_test_full_{v}.npy")) for v in ("plain", "hflip")], 0)
print("l75 s42 test identical to old run:", np.abs(old - T["l75"]).max())
