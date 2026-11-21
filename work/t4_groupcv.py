# Scene-group CV for T4 frozen heads. Why: train holds many near-duplicate scenes (train->train NN cosine median 0.75 vs
# test->train 0.66 on RADIO features), so random-fold OOF (0.96-0.99) measures memorised scenes, while the test set is new
# scenes of the same games. Grouping train images into scene clusters and holding whole clusters out gives a validation that
# behaves like the test set, so head choices can be ranked offline instead of on the LB. Train labels only.
# usage: python work/t4_groupcv.py            -> builds t4/research/groupcv/groups_k{K}.npy and prints a comparison table
import os, sys, time, numpy as np, pandas as pd, torch
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
torch.set_num_threads(os.cpu_count()); torch.manual_seed(0); np.random.seed(0)
OUT = "t4/research/groupcv"; os.makedirs(OUT, exist_ok=True)
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv")
C = sorted(tr["class"].unique()); y = tr["class"].map({c: i for i, c in enumerate(C)}).values; K = len(C)
SRC = {"radio": ("t4/preds/frozen_b/feats", "radio"), "vithp": ("t4/preds/dinov3_feats/feats", "vit_huge_plus_patch16_dinov3"),
       "pe": ("t4/preds/frozen_b/feats", "pe"), "siglip2": ("t4/preds/frozen_b/feats", "siglip2"),
       "dinov2g": ("t4/preds/frozen_c/feats", "dinov2g"), "radio3": ("t4/preds/frozen_c/feats", "radio3"),
       "radio3g": ("t4/preds/frozen_e/feats", "radio3g"), "webssl1b": ("t4/preds/frozen_e/feats", "webssl1b"), "aimv2h": ("t4/preds/frozen_e/feats", "aimv2h"),
       "vit7b": ("t4/preds/frozen_7b/feats", "vit_7b_patch16_dinov3"),
       **{f"vit7b_{t}": ("t4/preds/frozen_7b_ms/feats", f"vit7b_{t}") for t in ("s05", "s15", "l50", "l75")}}
def load(name, view="plain", split="train"):
    d, t = SRC[name]; f = f"{d}/{t}_{split}_{view}.npy"
    if not os.path.exists(f): f = f"t4/preds/frozen_d/feats/{t}_{split}_{view}.npy"     # Kernel D extra views
    return np.load(f).astype(np.float32)

def make_groups(kc, feat="radio"):
    p = f"{OUT}/groups_{feat}_k{kc}.npy"
    if os.path.exists(p): return np.load(p)
    X = torch.from_numpy(load(feat)); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    U, S, V = torch.pca_lowrank(X, q=128, niter=4); Z = torch.nn.functional.normalize(X @ V, dim=1).numpy()
    g = np.zeros(len(y), int)
    for c in range(K):
        m = y == c; g[m] = c * 1000 + KMeans(kc, n_init=3, random_state=0).fit_predict(Z[m])
    np.save(p, g); return g

def folds_group(g, n=5, seed=0):
    # GroupKFold is deterministic; shuffle group ids so different seeds give different splits
    rng = np.random.default_rng(seed); u = np.unique(g); perm = dict(zip(u, rng.permutation(len(u))))
    return list(GroupKFold(n).split(y, y, np.array([perm[x] for x in g])))

def fit_linear(Xs, Y, wd=1e-2, epochs=30, lr=1e-3, cw=True, ls=0.05):
    X = torch.cat(Xs); Yt = torch.from_numpy(np.tile(Y, len(Xs)))
    w = torch.tensor(1 / np.sqrt(np.bincount(Y, minlength=K) + 1), dtype=torch.float32); w = w / w.mean()
    m = torch.nn.Linear(X.shape[1], K); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); lf = torch.nn.CrossEntropyLoss(weight=w if cw else None, label_smoothing=ls)
    for _ in range(epochs):
        p = torch.randperm(len(X))
        for i in range(0, len(X), 1024):
            b = p[i:i+1024]; opt.zero_grad(); lf(m(X[b]), Yt[b]).backward(); opt.step()
        sch.step()
    return m

def cv(feats, folds, views=("plain", "hflip"), norm="std", part="all", **kw):
    """feats: list of backbone names (concatenated). Returns OOF proba."""
    def get(v):
        A = []
        for f in feats:
            a = load(f, v); d = a.shape[1] // 2
            a = a[:, :d] if part == "cls" else a[:, d:] if part == "avg" else a
            A.append(a)
        return np.concatenate(A, 1)
    X = {v: get(v) for v in views}; oof = np.zeros((len(y), K), np.float32)
    for a, b in folds:
        mu, sd = X["plain"][a].mean(0), X["plain"][a].std(0) + 1e-6
        def z(A):
            t = torch.from_numpy((A - mu) / sd)
            return torch.nn.functional.normalize(t, dim=1) * 30 if norm == "l2" else t
        m = fit_linear([z(X[v][a]) for v in views], y[a], **kw)
        ev = [v for v in ("plain", "hflip") if v in X]
        oof[b] = np.mean([torch.softmax(m(z(X[v][b])), 1).detach().numpy() for v in ev], 0)
    return oof

sc = lambda p, idx=None: (f1_score(y, p.argmax(1), average="macro"), accuracy_score(y, p.argmax(1)))
if __name__ == "__main__":
    t0 = time.time()
    rnd = list(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y))
    for kc in (40, 8):
        g = make_groups(kc); fg = folds_group(g)
        print(f"groups k={kc}/class: {len(np.unique(g))} groups, fold class mix ok:", [np.bincount(y[b], minlength=K).min() for _, b in fg], flush=True)
        for f in ("radio", "vithp"):
            a = sc(cv([f], rnd)); b = sc(cv([f], fg))
            print(f"  {f:8s} random-CV F1 {a[0]:.4f} acc {a[1]:.4f} | group-CV(k={kc}) F1 {b[0]:.4f} acc {b[1]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
