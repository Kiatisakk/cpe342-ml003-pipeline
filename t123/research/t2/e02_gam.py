"""E02: multinomial additive model (spline GAM via logistic regression).
Under class-conditional independence log p(k|x) = sum_j f_jk(x_j) + c_k exactly, so a multinomial
LR on per-feature spline bases is the correctly specified discriminative counterpart of NB.
usage: python t123/research/t2/e02_gam.py [n_knots] [C] [tag]"""
import sys; sys.path.insert(0, 't123/research/t2')
from runner import *
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LogisticRegression
from scipy import sparse

NK = int(sys.argv[1]) if len(sys.argv) > 1 else 12
C = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
TAG = sys.argv[3] if len(sys.argv) > 3 else f'gam_k{NK}_C{C}'
NUMD = [c for c in DISC if pd.api.types.is_numeric_dtype(tr[c])]


def design(a):
    """build design matrices for train rows (all N) and test rows, fitted on rows a only"""
    Xt, Xe = [], []
    for c in CONT:
        x = tr[c].values.astype(float); xe = te[c].values.astype(float)
        nt, ne = np.isnan(x), np.isnan(xe)
        st = SplineTransformer(n_knots=NK, degree=3, knots='quantile', extrapolation='constant')
        st.fit(x[a][~nt[a]].reshape(-1, 1))
        Bt = st.transform(np.where(nt, np.nanmedian(x[a]), x).reshape(-1, 1)); Bt[nt] = 0
        Be = st.transform(np.where(ne, np.nanmedian(x[a]), xe).reshape(-1, 1)); Be[ne] = 0
        lo, hi = np.nanmin(x[a]), np.nanmax(x[a])
        ext = lambda v, n: np.c_[n, (v <= lo) & ~n, (v >= hi) & ~n].astype(float)
        Xt += [Bt, ext(x, nt)]; Xe += [Be, ext(xe, ne)]
    for c in DISC:
        vt = tr[c].astype(object).where(tr[c].notna(), 'NA').astype(str).values
        ve = te[c].astype(object).where(te[c].notna(), 'NA').astype(str).values
        lv = sorted(set(vt[a]))
        Xt.append(np.stack([vt == l for l in lv], 1).astype(float)); Xe.append(np.stack([ve == l for l in lv], 1).astype(float))
    return np.hstack(Xt), np.hstack(Xe)


def fp(a, b):
    Xt, Xe = design(a)
    m = LogisticRegression(C=C, max_iter=3000, tol=1e-6)
    m.fit(Xt[a], y[a])
    return m.predict_proba(Xt[b]), m.predict_proba(Xe)


if __name__ == '__main__':
    run(TAG, fp)
