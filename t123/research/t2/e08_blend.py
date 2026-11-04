"""E08: summary of all saved models (raw argmax F1, nested-bias F1, logloss) + blends.
 - geo_<a+b>: equal-weight geometric mean of probabilities (no tuning)
 - stack: multinomial LR meta-learner on log-probas, nested (meta fit on the other 4 folds' OOF rows)
usage: python t123/research/t2/e08_blend.py model1 model2 ...  (names as saved in cache/P_<name>_s<seed>.npz)"""
import sys; sys.path.insert(0, 't123/research/t2')
from runner import *
from sklearn.linear_model import LogisticRegression
names = sys.argv[1:]
O = {n: load(n)[0] for n in names}


def line(name, oofs):
    raw = [f1m(y, o) for o in oofs]
    nb = [nested_bias_f1(np.log(np.clip(o, 1e-12, 1)), s)[0] for o, s in zip(oofs, SEEDS)]
    ll = [log_loss(y, np.clip(o, 1e-9, 1)) for o in oofs]
    print(f'{name:40s} raw {np.mean(raw):.4f}±{np.std(raw):.4f}  nested-bias {np.mean(nb):.4f}±{np.std(nb):.4f}  LL {np.mean(ll):.4f}', flush=True)


for n in names: line(n, O[n])
if len(names) > 1:
    G = softmax(np.mean([np.log(np.clip(O[n], 1e-12, 1)) for n in names], 0).reshape(-1, K)).reshape(len(SEEDS), N, K)
    line('geo(' + '+'.join(names) + ')', G)
    st = []
    for si, s in enumerate(SEEDS):
        X = np.concatenate([np.log(np.clip(O[n][si], 1e-12, 1)) for n in names], 1); P = np.zeros((N, K))
        for a, b in folds(s):
            P[b] = LogisticRegression(C=1.0, max_iter=1000).fit(X[a], y[a]).predict_proba(X[b])
        st.append(P)
    line('stackLR(' + '+'.join(names) + ')', np.array(st))
