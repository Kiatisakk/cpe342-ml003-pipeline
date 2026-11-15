# From-scratch MLP (no pretrained weights) predicting 11 quantiles of the offset target r = log y - log prev
# for payers, pinball loss, monotone quantiles (base + cumulative softplus increments). Hurdle median with 3-seed p.
# Scaling: QuantileTransformer(normal) fitted on the TRAINING fold only; NaN -> 0 after transform + missing flags.
# usage: python t3/research/exp_nnq.py NAME [--seeds 2] [--width 256] [--epochs 60]
import sys, os, time, argparse, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, save_oof, hurdle_median, extra_feats, prev_offset, NT
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--width", type=int, default=256); ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--drop", type=float, default=0.1); ap.add_argument("--wd", type=float, default=1e-5)
a = ap.parse_args(); torch.set_num_threads(NT)
ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof")
D = load(); X, Xt = extra_feats(D["X"], D["Xt"]); y, F = D["y"], D["folds"]; n = len(y)
P = np.mean([np.load(os.path.join(OD, f"{q}_p.npy")) for q in ("lgbh_log", "clf_s1", "clf_s2")], 0)
off, off_t = prev_offset(X), prev_offset(Xt)
X = X.assign(off=off); Xt = Xt.assign(off=off_t)
MISS = [c for c in X.columns if X[c].isna().any()]
AT = torch.tensor(ALPHAS, dtype=torch.float32)

class Net(nn.Module):
    def __init__(s, d):
        super().__init__(); L = []; w = d
        for _ in range(a.depth): L += [nn.Linear(w, a.width), nn.BatchNorm1d(a.width), nn.SiLU(), nn.Dropout(a.drop)]; w = a.width
        s.body = nn.Sequential(*L); s.head = nn.Linear(w, len(ALPHAS))
    def forward(s, x):
        h = s.head(s.body(x)); return torch.cat([h[:, :1], h[:, :1] + torch.cumsum(nn.functional.softplus(h[:, 1:]), 1)], 1)

def pinball(q, t):
    e = t[:, None] - q; return torch.maximum(AT * e, (AT - 1) * e).mean()

def prep(qt, Z):
    M = qt.transform(Z.values); M = np.nan_to_num(M, nan=0.0)
    return np.column_stack([M, Z[MISS].isna().values.astype(np.float32)]).astype(np.float32)

OQ = np.zeros((n, len(ALPHAS))); TQ = np.zeros((len(Xt), len(ALPHAS))); t0 = time.time()
for k, (tr, va) in enumerate(F):
    pos = tr[y[tr] > 0]
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200000, random_state=0).fit(X.iloc[pos].values)
    Xp = prep(qt, X.iloc[pos]); r = (np.log(y[pos]) - off[pos]).astype(np.float32)
    Xv, Xte = torch.tensor(prep(qt, X.iloc[va])), torch.tensor(prep(qt, Xt))
    for sd in range(a.seeds):
        torch.manual_seed(sd); np.random.seed(sd)
        i, j = train_test_split(np.arange(len(r)), test_size=0.1, random_state=sd)
        xi, ri, xj, rj = map(torch.tensor, (Xp[i], r[i], Xp[j], r[j]))
        net = Net(Xp.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((len(i) + 511) // 512))
        best, bstate, bad = 1e9, None, 0
        for ep in range(a.epochs):
            net.train(); perm = torch.randperm(len(i))
            for b in range(0, len(i), 512):
                idx = perm[b:b + 512]; opt.zero_grad(); loss = pinball(net(xi[idx]), ri[idx]); loss.backward(); opt.step(); sch.step()
            net.eval()
            with torch.no_grad(): v = pinball(net(xj), rj).item()
            if v < best - 1e-5: best, bstate, bad = v, {kk: vv.clone() for kk, vv in net.state_dict().items()}, 0
            else: bad += 1
            if bad >= 12: break
        net.load_state_dict(bstate); net.eval()
        with torch.no_grad():
            OQ[va] += net(Xv).numpy() / a.seeds          # average in residual-log space across seeds
            TQ += net(Xte).numpy() / a.seeds / 5
        print(f"fold {k} seed {sd} ep {ep} val pinball {best:.4f} {time.time()-t0:.0f}s", flush=True)
    OQ[va] = np.exp(np.sort(OQ[va], 1) + off[va][:, None])
TQ = np.exp(np.sort(TQ, 1) + off_t[:, None])
o = hurdle_median(P[:n], OQ, ALPHAS); t = hurdle_median(P[n:], TQ, ALPHAS)
save_oof(a.name, o, t); np.save(os.path.join(OD, f"{a.name}_Q.npy"), np.r_[OQ, TQ])
m, s = fold_scores(y, o, F)
print(f"RESULT {a.name}: OOF {nmae(y, o):.5f} fold mean {m:.5f} sd {s:.5f}")
