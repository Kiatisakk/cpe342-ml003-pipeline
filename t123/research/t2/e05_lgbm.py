"""E05: LightGBM multiclass. variants:
  raw   : raw features (noise removed), categoricals as codes fit on outer-train rows
  stump : additive model (max_depth=1) - a boosted GAM
  nbf   : raw features + honest inner-OOF NB per-feature log-lik ratios (GBM on NB features)
Early stopping on a 10% stratified split of the outer train rows (never the held-out fold).
usage: python t123/research/t2/e05_lgbm.py <variant>"""
import sys; sys.path.insert(0, 't123/research/t2')
from runner import *
import lightgbm as lgb
from sklearn.model_selection import train_test_split
V = sys.argv[1]
TH = int(os.environ.get('OMP_NUM_THREADS', 1))


def raw(a):
    Xt = tr[FEATS].copy(); Xe = te[FEATS].copy()
    for c in FEATS:
        if not pd.api.types.is_numeric_dtype(tr[c]):
            lv = {l: i for i, l in enumerate(sorted(tr[c].iloc[a].dropna().unique()))}
            Xt[c] = tr[c].map(lv).astype(float); Xe[c] = te[c].map(lv).astype(float)
    return Xt.values.astype(np.float32), Xe.values.astype(np.float32)


def fp(a, b):
    Xt, Xe = raw(a)
    if V == 'nbf':
        Lin, Lall, Lte = inner_blocks(a, seed=len(a) % 1000)
        f = lambda L: (L - L[:, :, :1]).reshape(len(L), -1)[:, [i for i in range(len(FEATS) * K) if i % K]]
        Fin = np.zeros((N, Xt.shape[1] + len(FEATS) * 3), np.float32)
        Fin[a] = np.c_[Xt[a], f(Lin)]; Fin[b] = np.c_[Xt[b], f(Lall[b])]
        Xt, Xe = Fin, np.c_[Xe, f(Lte)]
    p = dict(objective='multiclass', num_class=K, learning_rate=0.03, num_leaves=15, min_child_samples=100,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=TH)
    if V == 'stump': p.update(num_leaves=2, max_depth=1, learning_rate=0.1, feature_fraction=1.0)
    ia, iv = train_test_split(a, test_size=0.1, stratify=y[a], random_state=0)
    m = lgb.train(p, lgb.Dataset(Xt[ia], y[ia]), 5000, valid_sets=[lgb.Dataset(Xt[iv], y[iv])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict(Xt[b]), m.predict(Xe)


if __name__ == '__main__':
    run(f'lgbm_{V}', fp)
