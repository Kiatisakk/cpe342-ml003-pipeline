# Nested post-processing on a saved hurdle model (Q = positive-part quantiles, p = P(y>0)).
# Every parameter is tuned on 4 outer folds and applied to the 5th (same v4 folds) -> honest estimate.
# Methods:
#   A  global (p-shift, kappa)                      B  per-segment (p-shift, kappa)
#   C  global (quantile-level offset delta, kappa)  D  per-segment (delta, kappa)
#   E  per prediction-decile multiplicative scale (MAE-optimal = weighted median of y/pred, weights pred)
#   F  E per segment x 5 prediction bins            G  zero-out predictions below a threshold c
# usage: python t3/research/postproc.py NAME [--apply METHOD]   (NAME = model with oof/NAME_Q.npy, oof/NAME_p.npy)
import sys, os, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, save_oof

OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof")
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--apply", default=None); ap.add_argument("--out", default=None)
a = ap.parse_args()
D = load(); y, F = D["y"], D["folds"]; n = len(y)
Q = np.load(os.path.join(OD, f"{a.name}_Q.npy")); P = np.load(os.path.join(OD, f"{a.name}_p.npy"))
af = os.path.join(OD, f"{a.name}_alphas.npy")
AL = np.load(af) if os.path.exists(af) else np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
Q = np.sort(Q, 1); seg = np.r_[D["X"]["segment"].values, D["Xt"]["segment"].values].astype(int)


def pred(idx, shift=0.0, kap=1.0, delta=0.0):
    pp = np.clip(P[idx] + shift, 1e-6, 1)
    q = np.clip(1 - 1 / (2 * pp) + delta, AL[0], AL[-1])
    j = np.clip(np.searchsorted(AL, q) - 1, 0, len(AL) - 2); r = np.arange(len(q)); Qi = Q[idx]
    v = Qi[r, j] + (q - AL[j]) / (AL[j + 1] - AL[j]) * (Qi[r, j + 1] - Qi[r, j])
    return np.where(pp > 0.5, np.clip(kap * v, 0, 300000), 0.0)


SH = np.round(np.arange(-0.06, 0.0601, 0.01), 3); KA = np.round(np.arange(0.90, 1.1001, 0.01), 3); DL = np.round(np.arange(-0.06, 0.0601, 0.01), 3)
yy = np.r_[y, np.zeros(len(P) - n)]


def grid(idx, mode):
    best, bp = np.inf, None
    for s1 in (SH if mode == "shift" else DL):
        base = pred(idx, shift=s1) if mode == "shift" else pred(idx, delta=s1)
        for k in KA:
            e = np.abs(yy[idx] - k * base).sum()
            if e < best: best, bp = e, (s1, k)
    return bp


def apply_grid(idx, bp, mode):
    return pred(idx, shift=bp[0], kap=bp[1]) if mode == "shift" else pred(idx, delta=bp[0], kap=bp[1])


def wmedian(v, w):
    o = np.argsort(v); c = np.cumsum(w[o]); return v[o][np.searchsorted(c, c[-1] / 2)]


base_all = pred(np.arange(len(P)))


def fit_bins(idx, nb, by_seg):
    """bins on base prediction (>0 only), per bin scale = weighted median of y/pred (weights pred) -> MAE-optimal scale."""
    pos = idx[base_all[idx] > 0]; edges = np.quantile(base_all[pos], np.linspace(0, 1, nb + 1)[1:-1]); S = {}
    for g in (range(4) if by_seg else [None]):
        for b in range(nb):
            m = pos[(np.searchsorted(edges, base_all[pos]) == b) & ((seg[pos] == g) if by_seg else True)]
            S[(g, b)] = wmedian(yy[m] / base_all[m], base_all[m]) if len(m) > 50 else 1.0
    return edges, S


def apply_bins(idx, edges, S, by_seg):
    b = np.searchsorted(edges, base_all[idx]); g = seg[idx] if by_seg else [None] * len(idx)
    return base_all[idx] * np.array([S[(gi, bi)] for gi, bi in zip(g, b)])


def fit_zero(idx):
    cs = [0, 50, 100, 200, 300, 500, 800]; return min(cs, key=lambda c: np.abs(yy[idx] - np.where(base_all[idx] < c, 0, base_all[idx])).sum())


METHODS = {}
def method(name):
    def deco(f): METHODS[name] = f; return f
    return deco

@method("A")
def mA(tr, va): bp = grid(tr, "shift"); return apply_grid(va, bp, "shift"), bp
@method("B")
def mB(tr, va):
    o = np.zeros(len(va)); bps = {}
    for g in range(4):
        bp = grid(tr[seg[tr] == g], "shift"); m = seg[va] == g; o[m] = apply_grid(va[m], bp, "shift"); bps[g] = bp
    return o, bps
@method("C")
def mC(tr, va): bp = grid(tr, "delta"); return apply_grid(va, bp, "delta"), bp
@method("D")
def mD(tr, va):
    o = np.zeros(len(va)); bps = {}
    for g in range(4):
        bp = grid(tr[seg[tr] == g], "delta"); m = seg[va] == g; o[m] = apply_grid(va[m], bp, "delta"); bps[g] = bp
    return o, bps
@method("E")
def mE(tr, va): e, S = fit_bins(tr, 10, False); return apply_bins(va, e, S, False), np.round(list(S.values()), 3)
@method("F")
def mF(tr, va): e, S = fit_bins(tr, 5, True); return apply_bins(va, e, S, True), None
@method("G")
def mG(tr, va): c = fit_zero(tr); return np.where(base_all[va] < c, 0, base_all[va]), c


raw = base_all[:n]; m0, s0 = fold_scores(y, raw, F)
print(f"raw {a.name}: OOF {nmae(y, raw):.5f} fold {m0:.5f} +- {s0:.5f}")
for nm, f in METHODS.items():
    if a.apply and nm != a.apply: continue
    o = np.zeros(n); ps = []
    for tr, va in F:
        o[va], p_ = f(tr, va); ps.append(p_)
    d = np.array([nmae(y[v], o[v]) - nmae(y[v], raw[v]) for _, v in F]); m, s = fold_scores(y, o, F)
    print(f"RESULT {nm}: nested OOF {nmae(y, o):.5f} fold {m:.5f} +- {s:.5f} | paired gain {d.mean():+.5f} +- {d.std():.5f} | params fold0 {ps[0]}")
    if a.apply and a.out:   # final: tune on ALL OOF, apply to test
        alltr, te = np.arange(n), np.arange(n, len(P)); t, p_ = f(alltr, te)
        save_oof(a.out, o, t); print("test params", p_)
