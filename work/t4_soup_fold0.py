# Fold-0 check of the LoRA model soup (l75 + c30f10 val0) vs the single runs and the probability average; test agreement.
import numpy as np, pandas as pd, itertools
from sklearn.metrics import f1_score
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values; lg = lambda p: np.log(np.clip(p, 1e-9, 1))
A, Q3, Q4 = "t4/preds/v4f_lora7b/preds", "t4/preds/v4f_lora7b_q3", "t4/preds/v4f_lora7b_q4"
va = np.load(f"{A}/lora7b_l75_val0_val_idx.npy"); assert (np.load(f"{Q4}/lora7b_soup2_val0_val_idx.npy") == va).all(); yv = y[va]
pt = tr["class"].value_counts(normalize=True).reindex(C).values
f = lambda L: round(f1_score(yv, L.argmax(1), average="macro"), 5); err = lambda L: int((L.argmax(1) != yv).sum())
pv = lambda d, t: np.mean([lg(np.load(f"{d}/lora7b_{t}_oof_{v}.npy")) for v in ("plain", "hflip")], 0)
V = {"l75": pv(A, "l75_val0"), "c30f10": pv(Q3, "c30f10r16e3_val0"), "soup_avghead": pv(Q4, "soup2_val0_avghead"), "soup_refit": pv(Q4, "soup2_val0_refit")}
V["probavg"] = (V["l75"] + V["c30f10"]) / 2
for k, L in V.items(): print(f"{k:13s} F1 {f(L)} err {err(L)} | +train prior: F1 {f(L - np.log(pt))} err {err(L - np.log(pt))}")
print("fold0 agree", {f"{a}-{b}": round(float((V[a].argmax(1) == V[b].argmax(1)).mean()), 4) for a, b in itertools.combinations(V, 2)})
p = lambda d, t: np.mean([lg(np.load(f"{d}/v4f_lora7b_{t}_test_full_{v}.npy")) for v in ("plain", "hflip")], 0)
T = {"l75": p(A, "l75_s42"), "c30f10": p(Q3, "c30f10r16e3_s42"), "soup2": p(Q4, "soup2_s42"), "soup2avg": p(Q4, "soup2avghead_s42"),
     "soup3": p(Q4, "soup3_s42"), "soup3avg": p(Q4, "soup3avghead_s42")}
T["probavg"] = (T["l75"] + T["c30f10"]) / 2
print("test agree", {f"{a}-{b}": round(float((T[a].argmax(1) == T[b].argmax(1)).mean()), 4) for a, b in itertools.combinations(T, 2)})
print("test conf<.6", {k: round(float((np.exp(L - np.logaddexp.reduce(L, 1, keepdims=True)).max(1) < .6).mean()), 4) for k, L in T.items()})
