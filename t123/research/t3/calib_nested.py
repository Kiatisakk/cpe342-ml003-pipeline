# Nested per-group calibration of a hurdle model: per segment (shift on p, kappa scale) tuned on 4 outer folds,
# applied to the held-out fold. Final test params tuned on all OOF.
# usage: python t123/research/t3/calib_nested.py QNAME [--p PNAME] [--group seg|none] [--out NAME]
import sys, os, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, save_oof, hurdle_median

ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
SH = np.arange(-0.1, 0.1001, 0.02); KA = np.arange(0.86, 1.1401, 0.02)
OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof")


def tune(y, p, Q, idx):
    best, bp = -1, (0.0, 1.0)
    for sh in SH:
        base = hurdle_median(p[idx], Q[idx], ALPHAS, sh, 1.0)
        for ka in KA:
            s = -np.abs(y[idx] - ka * base).sum()
            if s > best: best, bp = s, (sh, ka)
    return bp


def run(qname, pname, group, out=None, verbose=True):
    D = load(); y, F, X, Xt = D["y"], D["folds"], D["X"], D["Xt"]; n = len(y)
    QQ = np.load(os.path.join(OD, f"{qname}_Q.npy")); PP = np.load(os.path.join(OD, f"{pname}_p.npy"))
    OQ, TQ, Op, Tp = QQ[:n], QQ[n:], PP[:n], PP[n:]
    g = X["segment"].values.astype(int) if group == "seg" else np.zeros(n, int)
    gt = Xt["segment"].values.astype(int) if group == "seg" else np.zeros(len(Xt), int)
    o = np.zeros(n)
    for tr, va in F:
        for s in np.unique(g):
            sh, ka = tune(y, Op, OQ, tr[g[tr] == s]); m = va[g[va] == s]
            o[m] = hurdle_median(Op[m], OQ[m], ALPHAS, sh, ka)
    t = np.zeros(len(Xt)); P = {}
    for s in np.unique(g):
        P[s] = tune(y, Op, OQ, np.where(g == s)[0]); m = gt == s
        t[m] = hurdle_median(Tp[m], TQ[m], ALPHAS, *P[s])
    raw = hurdle_median(Op, OQ, ALPHAS)
    mm, sd = fold_scores(y, o, F); m0, s0 = fold_scores(y, raw, F)
    if verbose:
        print(f"RESULT calib {qname}/{pname}/{group}: raw {nmae(y, raw):.5f} -> nested {nmae(y, o):.5f} (fold {mm:.5f} +- {sd:.5f}) params {P}")
    if out: save_oof(out, o, t)
    return o, t


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("q"); ap.add_argument("--p", default="lgbh_log")
    ap.add_argument("--group", default="seg"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); run(a.q, a.p, a.group, a.out)
