# Low-memory frozen linear head (features stay float16; float32 only per batch). Train labels only.
# Same head as work/t4_dinov3_head.py (softmax regression, class weights 1/sqrt(n), label smoothing .05, AdamW, cosine)
# but features are standardised once with train-plain statistics and kept in float16.
# usage: python work/t4_head_lowmem.py <feats dir> <backbone tag> <train views, e.g. plain,hflip> <out suffix> [--groupcv]
#   --groupcv : also score scene-group CV (sig8 / sig30 folds from work/t4_groupcv.py, cached groups) -> printed
# Writes t4/preds/dinov3_head/<tag><suffix>/d3lin_<tag><suffix>_{oof,test_fold{k}}.npy (test in OLD order, mean plain+hflip).
import sys, os, numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
torch.set_num_threads(os.cpu_count())
D, tag, views, suf = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]; GCV = "--groupcv" in sys.argv
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv")
C = sorted(tr["class"].unique()); y = tr["class"].map({c: i for i, c in enumerate(C)}).values; K = len(C)
assert (np.load(f"{D}/train_names.npy", allow_pickle=True) == tr["id"].astype(str).str.replace(".jpg", "").values).all()
ten = np.load(f"{D}/test_names.npy", allow_pickle=True); ref = pd.read_csv("t4/preds/v3_5f/preds/task4_v3_5f.csv")["id_img"].astype(str).values
pos = pd.Series(np.arange(len(ten)), index=ten).loc[ref].values
ld = lambda s, v: np.load(f"{D}/{tag}_{s}_{v}.npy", mmap_mode="r")
P = ld("train", "plain"); n, d = P.shape; s1 = np.zeros(d); s2 = np.zeros(d)
for i in range(0, n, 4096): c = P[i:i+4096].astype(np.float64); s1 += c.sum(0); s2 += (c * c).sum(0)
mu = (s1 / n).astype(np.float32); sd = (np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0)) + 1e-6).astype(np.float32); del P
def std16(A):
    out = np.empty(A.shape, np.float16)
    for i in range(0, len(A), 4096): out[i:i+4096] = (A[i:i+4096].astype(np.float32) - mu) / sd
    return out
Xtr = {v: std16(ld("train", v)) for v in views}
Xte = {v: std16(ld("test", v))[pos] for v in ("plain", "hflip")}
print(tag, "dim", d, "views", views, flush=True)

def fit(idx, wd, epochs=40):
    Y = torch.from_numpy(np.tile(y[idx], len(views))); X = [torch.from_numpy(Xtr[v][idx]) for v in views]; X = torch.cat(X)
    cw = torch.tensor(1 / np.sqrt(np.bincount(y[idx], minlength=K)), dtype=torch.float32); cw = cw / cw.mean()
    m = torch.nn.Linear(d, K); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); lf = torch.nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    for _ in range(epochs):
        p = torch.randperm(len(X))
        for i in range(0, len(X), 1024):
            b = p[i:i+1024]; opt.zero_grad(); lf(m(X[b].float()), Y[b]).backward(); opt.step()
        sch.step()
    del X; return m
@torch.no_grad()
def pred(m, A, idx=None):
    A = A if idx is None else A[idx]
    return np.concatenate([torch.softmax(m(torch.from_numpy(A[i:i+8192]).float()), 1).numpy() for i in range(0, len(A), 8192)])
ev = lambda m, idx: (pred(m, Xtr["plain"], idx) + pred(m, Xtr["hflip"], idx)) / 2

folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y))
a, b = folds[0]; ia, ib = a[: int(0.8 * len(a))], a[int(0.8 * len(a)):]
best = max([1e-4, 1e-2, 1e-1], key=lambda wd: f1_score(y[ib], ev(fit(ia, wd, 20), ib).argmax(1), average="macro"))
print("weight decay", best, flush=True)
if GCV:
    sys.path.insert(0, "work"); import t4_groupcv as G
    for gn, kc in (("sig8", 8), ("sig30", 30)):
        oof = np.zeros((len(y), K), np.float32)
        for a, b in G.folds_group(G.make_groups(kc, "siglip2")):
            m = fit(a, 1e-2, 30)                                  # group-CV setting of work/t4_groupcv.py (wd 1e-2, 30 ep)
            oof[b] = ev(m, b)
        os.makedirs(f"{G.OUT}/oof", exist_ok=True); np.save(f"{G.OUT}/oof/{tag}{suf}__{gn}.npy", oof)
        print(f"groupCV {gn} macroF1 {f1_score(y, oof.argmax(1), average='macro'):.4f}", flush=True)
run = f"d3lin_{tag}{suf}"; OUT = f"t4/preds/dinov3_head/{tag}{suf}"; os.makedirs(OUT, exist_ok=True)
oof = np.zeros((len(y), K), np.float32)
for k, (a, b) in enumerate(folds):
    m = fit(a, best); oof[b] = ev(m, b)
    np.save(f"{OUT}/{run}_test_fold{k}.npy", (pred(m, Xte["plain"]) + pred(m, Xte["hflip"])) / 2)
np.save(f"{OUT}/{run}_oof.npy", oof)
print(f"{run} random-CV OOF macroF1 {f1_score(y, oof.argmax(1), average='macro'):.5f} -> {OUT}", flush=True)
