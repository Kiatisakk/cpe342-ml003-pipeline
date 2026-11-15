# Transductive kNN smoothing, validated on scene-group folds (train labels only for scoring).
# Idea: held-out scenes (like the test set) contain many near-duplicate frames; averaging the blend's log-probs over each
# image's nearest neighbours INSIDE the held-out set (frozen-feature cosine) lets confident frames fix their noisy twins.
# No model is trained on the held-out/test rows; only inference-time neighbours are used.
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "work"); import t4_groupcv as G
from sklearn.metrics import f1_score
y = G.y; FOLD = np.load(f"{G.OUT}/sgkf_sig8_fold.npy"); lg = lambda p: np.log(p + 1e-9)
ft = np.load("t4/preds/scenecv_s/preds/v4g_cnxt_s_d3_base_oof.npy")
F_ = {h: np.load(f"{G.OUT}/oof_sgkf/{h}.npy") for h in ["vithp", "radio", "radio3", "dinov2g"]}
fz = .4 * lg(F_["vithp"]) + .1 * lg(F_["radio"]) + .2 * lg(F_["radio3"]) + .3 * lg(F_["dinov2g"])
L = .5 * lg(ft) + .5 * fz; m = ~np.isnan(ft[:, 0])
def emb(names, idx, ref):                      # standardise with train-fold stats (ref), PCA-free cosine
    Z = []
    for n in names:
        X = G.load(n); mu, sd = X[ref].mean(0), X[ref].std(0) + 1e-6
        Z.append(torch.nn.functional.normalize(torch.from_numpy((X[idx] - mu) / sd), dim=1))
    return torch.nn.functional.normalize(torch.cat(Z, 1), dim=1)
def smooth(Lb, Z, k, lam, tau=None):
    S = Z @ Z.T; S.fill_diagonal_(-1); v, i = S.topk(k, 1)
    w = torch.softmax(v / tau, 1) if tau else torch.full_like(v, 1 / k)
    P = torch.from_numpy(Lb).float()
    return ((1 - lam) * P + lam * (w.unsqueeze(2) * P[i]).sum(1)).numpy(), v[:, 0].median().item()
f1 = lambda A: f1_score(y[m], A[m].argmax(1), average="macro")
print("baseline FT50+frozen4:", round(f1(L), 4), "| FT only", round(f1(lg(ft)), 4), "| frozen4 only", round(f1(fz), 4), flush=True)
for names in (["vithp"], ["radio3"], ["vithp", "radio3", "dinov2g"]):
    Zs = {k: emb(names, np.where(FOLD == k)[0], np.where(FOLD != k)[0]) for k in range(3)}
    for kk, lam, tau in [(5, .5, None), (10, .5, None), (20, .5, None), (10, .3, None), (10, .7, None), (10, 1.0, None), (20, .7, .05), (50, .7, .05), (50, .7, .1)]:
        out = np.full_like(L, np.nan); nn = []
        for k in range(3):
            b = np.where(FOLD == k)[0]; out[b], q = smooth(L[b], Zs[k], kk, lam, tau); nn.append(q)
        print(f"{'+'.join(names):22s} k={kk:3d} lam={lam} tau={tau}: F1 {f1(out):.4f} per-fold {[round(f1_score(y[FOLD==k], out[FOLD==k].argmax(1), average='macro'),4) for k in range(3)]} NN1 cos {np.mean(nn):.2f}", flush=True)
# does the test set have the same near-duplicate structure?
for n in ["vithp"]:
    X = G.load(n); mu, sd = X.mean(0), X.std(0) + 1e-6; T = G.load(n, split="test")
    Zt = torch.nn.functional.normalize(torch.from_numpy((T - mu) / sd), dim=1); r = torch.randperm(len(Zt))[:4000]
    S = Zt[r] @ Zt.T; S[torch.arange(len(r)), r] = -1; print("test->test NN1 cosine median", S.max(1).values.median().item(), "10th NN", S.topk(10, 1).values[:, -1].median().item())
    b = np.where(FOLD == 0)[0]; Zb = torch.nn.functional.normalize(torch.from_numpy((X[b] - mu) / sd), dim=1); S = Zb @ Zb.T; S.fill_diagonal_(-1)
    print("held-out fold0 NN1 cosine median", S.max(1).values.median().item(), "10th NN", S.topk(10, 1).values[:, -1].median().item())
