# Frozen linear heads on the SAME scene folds as the fine-tuned scene-CV kernel (t4/research/groupcv/sgkf_sig8_fold.npy, folds 0-2),
# so fine-tuned vs frozen blend weights can be fitted offline. Saves OOF (NaN = fold not run) to t4/research/groupcv/oof_sgkf/<head>.npy
import os, sys, numpy as np, torch
sys.path.insert(0, "work"); import t4_groupcv as G
y, K = G.y, G.K; FOLD = np.load(f"{G.OUT}/sgkf_sig8_fold.npy"); os.makedirs(f"{G.OUT}/oof_sgkf", exist_ok=True)
for name in sys.argv[1:]:
    X = {v: G.load(name, v) for v in ("plain", "hflip")}; oof = np.full((len(y), K), np.nan, np.float32)
    for k in (0, 1, 2):
        a, b = np.where(FOLD != k)[0], np.where(FOLD == k)[0]
        mu, sd = X["plain"][a].mean(0), X["plain"][a].std(0) + 1e-6; z = lambda A: torch.from_numpy((A - mu) / sd)
        m = G.fit_linear([z(X[v][a]) for v in X], y[a])
        oof[b] = np.mean([torch.softmax(m(z(X[v][b])), 1).detach().numpy() for v in X], 0)
    np.save(f"{G.OUT}/oof_sgkf/{name}.npy", oof); m_ = ~np.isnan(oof[:, 0])
    from sklearn.metrics import f1_score
    print(name, "rows", m_.sum(), "F1", round(f1_score(y[m_], oof[m_].argmax(1), average="macro"), 4), flush=True)
