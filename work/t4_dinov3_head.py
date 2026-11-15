# Linear head on frozen DINOv3 features (from Kaggle kernel task4-5fold-ensemble v7+, t4/preds/dinov3_feats/feats).
# usage: python work/t4_dinov3_head.py <backbone tag, e.g. vit_huge_plus_patch16_dinov3> [views=plain,hflip,dark,sat]
# Train labels only. 5-fold StratifiedKFold (seed 42); each fold trains on all train views of the train rows, predicts the
# val rows (mean of plain+hflip) and the test set (mean of plain+hflip). Softmax regression in torch (CPU), standardised
# features, weight decay chosen on fold 0's inner split.
# Output in the blend-script format, test rows in the OLD test order (t4/preds/v3_5f/preds/task4_v3_5f.csv id_img):
#   t4/preds/dinov3_head/<tag>/<run>_oof.npy, <run>_test_fold{k}.npy, run = d3lin_<short>
import sys, os, numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
torch.set_num_threads(os.cpu_count())
tag = sys.argv[1]; views = (sys.argv[2] if len(sys.argv) > 2 else "plain,hflip,dark,sat").split(",")
D = os.environ.get("T4_FEATS_DIR", "t4/preds/dinov3_feats/feats")
# env T4_TEST_VIEWS (default plain,hflip; probabilities averaged over views), T4_SUFFIX (output tag suffix, e.g. _v7tta).
# Views missing in T4_FEATS_DIR are read from t4/preds/frozen_d/feats (Kernel D: train dark2/sat2/darksat, test bright/desat/brdesat).
TEST_VIEWS = os.environ.get("T4_TEST_VIEWS", "plain,hflip").split(",")
short = {"vit_large_patch16_dinov3": "vitl", "vit_huge_plus_patch16_dinov3": "vithp"}.get(tag, tag) + os.environ.get("T4_SUFFIX", ""); run = f"d3lin_{short}"
OUT = f"t4/preds/dinov3_head/{short}"; os.makedirs(OUT, exist_ok=True)
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv")
C = sorted(tr["class"].unique()); y = tr["class"].map({c: i for i, c in enumerate(C)}).values; K = len(C)
trn = np.load(f"{D}/train_names.npy", allow_pickle=True); assert (trn == tr["id"].astype(str).str.replace(".jpg", "").values).all()
ten = np.load(f"{D}/test_names.npy", allow_pickle=True)
ref = pd.read_csv("t4/preds/v3_5f/preds/task4_v3_5f.csv")["id_img"].astype(str).values
pos = pd.Series(np.arange(len(ten)), index=ten).loc[ref].values                       # feature row for each old-order test row
def L(split, v):
    f = f"{D}/{tag}_{split}_{v}.npy"
    return np.load(f if os.path.exists(f) else f"t4/preds/frozen_d/feats/{tag}_{split}_{v}.npy").astype(np.float32)
Xtr = {v: L("train", v) for v in views}
Xte = {v: L("test", v)[pos] for v in TEST_VIEWS}
Xev = (Xtr["plain"] + Xtr["hflip"]) / 2 if "hflip" in Xtr else Xtr["plain"]
mu, sd = Xtr["plain"].mean(0), Xtr["plain"].std(0) + 1e-6
z = lambda a: torch.from_numpy((a - mu) / sd)
print(tag, {v: Xtr[v].shape for v in Xtr}, "test views", TEST_VIEWS, flush=True)

def fit(idx, wd, epochs=40):
    X = torch.cat([z(Xtr[v][idx]) for v in views]); Y = torch.from_numpy(np.tile(y[idx], len(views)))
    cw = torch.tensor(1 / np.sqrt(np.bincount(y[idx], minlength=K)), dtype=torch.float32); cw = cw / cw.mean()
    m = torch.nn.Linear(X.shape[1], K); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); lossf = torch.nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    for _ in range(epochs):
        p = torch.randperm(len(X))
        for i in range(0, len(X), 1024):
            b = p[i:i+1024]; opt.zero_grad(); lossf(m(X[b]), Y[b]).backward(); opt.step()
        sch.step()
    return m
pred = lambda m, A: torch.softmax(m(z(A)), 1).detach().numpy()

folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y))
a, b = folds[0]; ia, ib = a[: int(0.8 * len(a))], a[int(0.8 * len(a)):]
best = max([1e-4, 1e-2, 1e-1], key=lambda wd: f1_score(y[ib], pred(fit(ia, wd, 20), Xev[ib]).argmax(1), average="macro"))
print("weight decay", best, flush=True)
oof = np.zeros((len(y), K), np.float32)
for k, (a, b) in enumerate(folds):
    m = fit(a, best); oof[b] = pred(m, Xev[b]); pv = {v: pred(m, Xte[v]) for v in TEST_VIEWS}; np.save(f"{OUT}/{run}_test_fold{k}.npy", np.mean(list(pv.values()), 0))
    if len(TEST_VIEWS) > 2: [np.save(f"{OUT}/{run}_test_fold{k}_{v}.npy", pv[v]) for v in TEST_VIEWS]      # per-view, to compose TTA offline
    print(f"fold {k} macroF1 {f1_score(y[b], oof[b].argmax(1), average='macro'):.4f}", flush=True)
np.save(f"{OUT}/{run}_oof.npy", oof)
print(f"{run} OOF macroF1 {f1_score(y, oof.argmax(1), average='macro'):.5f} -> {OUT}")
