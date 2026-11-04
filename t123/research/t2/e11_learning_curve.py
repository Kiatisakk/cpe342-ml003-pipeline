"""E11: learning curve of baseline NB (and EM-NB). Train on a stratified fraction of each outer-train fold,
evaluate on the full held-out fold. If F1 is flat as n grows, density estimation is saturated -> Bayes ceiling."""
import sys; sys.argv = sys.argv[:1] + ['15', 'opt', 'em', '1.0']
sys.path.insert(0, 't123/research/t2')
from runner import *
import e06_emnb as E
from sklearn.model_selection import train_test_split
res = {}
for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
    for mode in ['nb', 'em']:
        sc = []
        for s in SEEDS:
            pred = np.zeros(N, int)
            for a, b in folds(s):
                aa = a if frac == 1 else train_test_split(a, train_size=frac, stratify=y[a], random_state=s)[0]
                B = E.binned(aa, E.CFG)
                if mode == 'nb':
                    tabs = E.densities(B, aa, np.eye(K)[y[aa]], 1.0); pi = np.bincount(y[aa], minlength=K) / len(aa)
                else:
                    tabs, pi, _ = E.fit(aa, B)
                pred[b] = (E.loglik(B, tabs, b) + np.log(pi)).argmax(1)
            sc.append(f1_score(y, pred, average='macro'))
        res[(frac, mode)] = report(f'{mode} frac {frac} (n_train≈{int(frac*0.8*N)})', sc)
