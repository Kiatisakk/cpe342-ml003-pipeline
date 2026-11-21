# One MLP head on the concatenated frozen features of the 4 backbones (ViT-H+, RADIOv4, RADIOv3, DINOv2-g).
# Strict-clean: train views plain + hflip only, train labels only, random 5-fold CV (seed 42), fixed hyper-parameters.
# Features stay float16 (standardised with train-plain statistics); float32 only per batch.
# Output t4/preds/dinov3_head/concat4_mlp/d3lin_concat4_mlp_{oof,test_fold{k}}.npy, test rows in the OLD order.
import os, numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
torch.set_num_threads(os.cpu_count()); torch.manual_seed(0)
ALL = {"vithp": ("t4/preds/dinov3_feats/feats", "vit_huge_plus_patch16_dinov3"), "radio": ("t4/preds/frozen_b/feats", "radio"),
       "radio3": ("t4/preds/frozen_c/feats", "radio3"), "dinov2g": ("t4/preds/frozen_c/feats", "dinov2g"),
       "radio3g": ("t4/preds/frozen_e/feats", "radio3g"), "webssl1b": ("t4/preds/frozen_e/feats", "webssl1b"),
       "siglip2": ("t4/preds/frozen_b/feats", "siglip2"), "pe": ("t4/preds/frozen_b/feats", "pe"),
       "vit7b": ("t4/preds/frozen_7b/feats", "vit_7b_patch16_dinov3")}
NAMES = os.environ.get("T4_CONCAT", "vithp,radio,radio3,dinov2g").split(",")       # backbones to concatenate
SRC = [ALL[n] for n in NAMES]; TAG = os.environ.get("T4_MLP_TAG", "concat4_mlp"); torch.manual_seed(int(os.environ.get("T4_SEED", "0")))
HID, DROP_IN, DROP, EPOCHS, LR, WD = 512, 0.2, 0.3, 20, 1e-3, 0.05
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv")
C = sorted(tr["class"].unique()); y = tr["class"].map({c: i for i, c in enumerate(C)}).values; K = len(C)
ref = pd.read_csv("t4/preds/v3_5f/preds/task4_v3_5f.csv")["id_img"].astype(str).values
def std16(A, mu, sd):
    out = np.empty(A.shape, np.float16)
    for i in range(0, len(A), 4096): out[i:i+4096] = (A[i:i+4096].astype(np.float32) - mu) / sd
    return out
Xtr = {"plain": [], "hflip": []}; Xte = {"plain": [], "hflip": []}
for D, t in SRC:
    assert (np.load(f"{D}/train_names.npy", allow_pickle=True) == tr["id"].astype(str).str.replace(".jpg", "").values).all()
    ten = np.load(f"{D}/test_names.npy", allow_pickle=True); pos = pd.Series(np.arange(len(ten)), index=ten).loc[ref].values
    P = np.load(f"{D}/{t}_train_plain.npy", mmap_mode="r"); mu = np.zeros(P.shape[1]); s2 = np.zeros(P.shape[1])
    for i in range(0, len(P), 4096): c = P[i:i+4096].astype(np.float64); mu += c.sum(0); s2 += (c * c).sum(0)
    mu /= len(P); sd = (np.sqrt(np.maximum(s2 / len(P) - mu ** 2, 0)) + 1e-6).astype(np.float32); mu = mu.astype(np.float32)
    for v in ("plain", "hflip"):
        Xtr[v].append(std16(np.load(f"{D}/{t}_train_{v}.npy", mmap_mode="r"), mu, sd))
        Xte[v].append(std16(np.load(f"{D}/{t}_test_{v}.npy", mmap_mode="r"), mu, sd)[pos])
    print(t, P.shape, flush=True)
Xtr = {v: np.concatenate(a, 1) for v, a in Xtr.items()}; Xte = {v: np.concatenate(a, 1) for v, a in Xte.items()}
d = Xtr["plain"].shape[1]; print("concat dim", d, flush=True)

def net(): return torch.nn.Sequential(torch.nn.Dropout(DROP_IN), torch.nn.Linear(d, HID), torch.nn.GELU(), torch.nn.Dropout(DROP), torch.nn.Linear(HID, K))
def fit(idx):
    X = torch.cat([torch.from_numpy(Xtr[v][idx]) for v in ("plain", "hflip")]); Y = torch.from_numpy(np.tile(y[idx], 2))
    cw = torch.tensor(1 / np.sqrt(np.bincount(y[idx], minlength=K)), dtype=torch.float32); cw = cw / cw.mean()
    m = net(); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD); steps = EPOCHS * ((len(X) + 511) // 512)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1); lf = torch.nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    for _ in range(EPOCHS):
        m.train(); p = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            b = p[i:i+512]; opt.zero_grad(); lf(m(X[b].float()), Y[b]).backward(); opt.step(); sch.step()
    return m.eval()
@torch.no_grad()
def pred(m, A, idx=None):
    A = A if idx is None else A[idx]
    return np.concatenate([torch.softmax(m(torch.from_numpy(A[i:i+4096]).float()), 1).numpy() for i in range(0, len(A), 4096)])
if os.environ.get("T4_GROUPCV"):   # scene-group CV only (sig8 / sig30 folds of work/t4_groupcv.py), same fixed head; no test output
    import sys; sys.path.insert(0, "work"); import t4_groupcv as G
    for gn, kc in (("sig8", 8), ("sig30", 30)):
        oof = np.zeros((len(y), K), np.float32)
        for a, b in G.folds_group(G.make_groups(kc, "siglip2")):
            m = fit(a); oof[b] = (pred(m, Xtr["plain"], b) + pred(m, Xtr["hflip"], b)) / 2
        os.makedirs(f"{G.OUT}/oof", exist_ok=True); np.save(f"{G.OUT}/oof/{TAG}__{gn}.npy", oof)
        print(f"{TAG} groupCV {gn} macroF1 {f1_score(y, oof.argmax(1), average='macro'):.4f}", flush=True)
    sys.exit(0)
run = f"d3lin_{TAG}"; OUT = f"t4/preds/dinov3_head/{TAG}"; os.makedirs(OUT, exist_ok=True)
oof = np.zeros((len(y), K), np.float32)
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
    m = fit(a); oof[b] = (pred(m, Xtr["plain"], b) + pred(m, Xtr["hflip"], b)) / 2
    np.save(f"{OUT}/{run}_test_fold{k}.npy", (pred(m, Xte["plain"]) + pred(m, Xte["hflip"])) / 2)
    print(f"fold {k} macroF1 {f1_score(y[b], oof[b].argmax(1), average='macro'):.4f}", flush=True)
np.save(f"{OUT}/{run}_oof.npy", oof)
print(f"{run} random-CV OOF macroF1 {f1_score(y, oof.argmax(1), average='macro'):.5f} -> {OUT}")
