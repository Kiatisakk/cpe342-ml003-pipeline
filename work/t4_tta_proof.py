# Is multi-scale TTA of the l75 LoRA real on scene fold 0? Errors with/without train prior, McNemar exact test, scene bootstrap.
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values; lg = lambda p: np.log(np.clip(p, 1e-9, 1))
pt = tr["class"].value_counts(normalize=True).reindex(C).values
D = "t4/preds/v4f_lora7b_q3"; va = np.load("t4/preds/v4f_lora7b/preds/lora7b_l75_val0_val_idx.npy"); yv = y[va]
G = np.load("t4/research/groupcv/sgkf_sig8_group.npy")[va]
S = ["224x416", "192x352", "256x480", "288x528"]; P = {s: lg(np.load(f"{D}/lora7b_l75_val0_tta_{s}.npy")) for s in S}
base, tta = P[S[0]], np.mean([P[s] for s in S], 0)
for pri in (False, True):
    b, t = (base - np.log(pt), tta - np.log(pt)) if pri else (base, tta)
    eb, et = b.argmax(1) != yv, t.argmax(1) != yv
    n01, n10 = int((~eb & et).sum()), int((eb & ~et).sum())   # TTA breaks / TTA fixes
    p = binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1
    print(f"{'train prior' if pri else 'raw':11s}: errors {eb.sum()} -> {et.sum()} | F1 {f1_score(yv, b.argmax(1), average='macro'):.5f} -> {f1_score(yv, t.argmax(1), average='macro'):.5f}"
          f" | TTA fixes {n10}, breaks {n01}, McNemar p = {p:.3f}")
    # scene bootstrap (resample scene groups): how often is TTA at least as good?
    ug = np.unique(G); rng = np.random.default_rng(0); d = []
    idx = {g: np.where(G == g)[0] for g in ug}
    for _ in range(2000):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]); d.append(int(eb[s].sum()) - int(et[s].sum()))
    d = np.array(d); print(f"{'':11s}  scene bootstrap: TTA fewer errors in {np.mean(d > 0):.0%}, equal {np.mean(d == 0):.0%}, more {np.mean(d < 0):.0%}")
