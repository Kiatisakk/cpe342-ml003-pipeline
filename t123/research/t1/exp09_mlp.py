"""From-scratch MLP (PyTorch, CPU) with piecewise-linear encoding (PLE) of numeric features.

Per fold: PLE bin edges = quantiles of the training-fold rows only; NaN -> all-zero encoding + missing flag.
Early stopping on the same inner 10% split scheme as the GBMs. No pretrained weights anywhere.
usage: python exp09_mlp.py [seeds...]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import torch, torch.nn as nn

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 2)))


def ple_fit(X, n_bins=16):
    edges = []
    for j in range(X.shape[1]):
        v = X[:, j]; v = v[~np.isnan(v)]
        e = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
        edges.append(e if len(e) > 1 else np.array([v.min(), v.min() + 1]))
    return edges


def ple_transform(X, edges):
    out = []
    for j, e in enumerate(edges):
        x = X[:, j]; nan = np.isnan(x); xf = np.where(nan, e[0], x)
        lo, hi = e[:-1][None, :], e[1:][None, :]
        enc = np.clip((xf[:, None] - lo) / np.maximum(hi - lo, 1e-12), 0, 1)
        enc[nan] = 0
        out += [enc, nan[:, None].astype(float)]
    return np.hstack(out).astype(np.float32)


class Net(nn.Module):
    def __init__(self, d, h=256, drop=0.25, depth=3):
        super().__init__()
        L, din = [], d
        for _ in range(depth):
            L += [nn.Linear(din, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(drop)]; din = h
        self.body = nn.Sequential(*L); self.head = nn.Linear(h, 1)
    def forward(self, x): return self.head(self.body(x)).squeeze(-1)


def train_mlp(Xa, ya, Xv, yv, seed, epochs=40, bs=512, lr=2e-3, wd=1e-4, h=256, drop=0.25, depth=3, patience=6):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(Xa.shape[1], h, drop, depth); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(Xa) / bs)))
    Xa_t, ya_t, Xv_t = torch.from_numpy(Xa), torch.from_numpy(ya.astype(np.float32)), torch.from_numpy(Xv)
    lossf = nn.BCEWithLogitsLoss(); best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2: continue
            opt.zero_grad(); loss = lossf(net(Xa_t[idx]), ya_t[idx]); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad(): pv = net(Xv_t).numpy()
        auc = roc_auc_score(yv, pv)
        if auc > best: best, bad, best_state = auc, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(best_state); net.eval()
    return net


def predict(net, X):
    with torch.no_grad(): return torch.sigmoid(net(torch.from_numpy(X))).numpy()


def mlp_fp(**kw):
    def f(Xa, ya, Xv, yv, Xb, Xt, rs):
        edges = ple_fit(Xa.values)
        T = lambda D: ple_transform(D.values, edges)
        net = train_mlp(T(Xa), ya, T(Xv), yv, rs, **kw)
        return predict(net, T(Xb)), (predict(net, T(Xt)) if Xt is not None else None)
    return f


if __name__ == "__main__":
    lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
    X = fe_v2(lab, pd.concat([lab, unl])).astype(float)
    seeds = tuple(int(s) for s in sys.argv[1:]) or SEEDS
    a, o = multi_seed("exp09 mlp PLE16 3x256 fe_v2", y, lambda s: cv_generic(mlp_fp(), X, y, seed=s)[0], seeds=seeds, save=len(seeds) == 3)
    if len(seeds) == 3: np.save(f"{OUT}/exp09_mlp.npy", np.array(o))
