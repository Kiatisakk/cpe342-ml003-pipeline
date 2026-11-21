# Battery of frozen-head variants scored with scene-group CV (see work/t4_groupcv.py). Train labels only.
# Groups come from a backbone that is NOT being scored where possible (siglip2 / pe), so no head is penalised by its own clusters.
# usage: python work/t4_groupcv_exp.py <exp name> ...   (no args = default list). Appends to t4/research/groupcv/results.csv,
# saves OOF proba to t4/research/groupcv/oof/<exp>__<groups>.npy
import os, sys, time, json, numpy as np, pandas as pd, torch
sys.path.insert(0, "work"); import t4_groupcv as G
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
y, K = G.y, G.K; os.makedirs(f"{G.OUT}/oof", exist_ok=True)
GROUPS = {"sig8": (8, "siglip2"), "sig30": (30, "siglip2"), "pe8": (8, "pe")}
FOLDS = {n: G.folds_group(G.make_groups(k, f)) for n, (k, f) in GROUPS.items()}

def feats(names, view, part="all"):
    A = []
    for f in names:
        a = G.load(f, view); d = a.shape[1] // 2
        A.append(a[:, :d] if part == "cls" else a[:, d:] if part == "avg" else a)
    return np.concatenate(A, 1) if len(A) > 1 else A[0]

def run(exp, names, views=("plain", "hflip"), part="all", norm="std", head="lin", groups=("sig8", "sig30"), **kw):
    X = {v: feats(names, v, part) for v in views}; out = {}
    for gn in groups:
        t0 = time.time(); oof = np.zeros((len(y), K), np.float32)
        for a, b in FOLDS[gn]:
            mu, sd = X["plain"][a].mean(0), X["plain"][a].std(0) + 1e-6
            def z(A):
                t = torch.from_numpy((A - mu) / sd)
                return torch.nn.functional.normalize(t, dim=1) * 30 if norm == "l2" else t
            ev = [v for v in ("plain", "hflip") if v in X]
            if head == "lin":
                m = G.fit_linear([z(X[v][a]) for v in views], y[a], **kw)
                oof[b] = np.mean([torch.softmax(m(z(X[v][b])), 1).detach().numpy() for v in ev], 0)
            elif head == "ncm":                                    # cosine nearest class mean
                Z = torch.nn.functional.normalize(torch.cat([z(X[v][a]) for v in views]), dim=1); Y = np.tile(y[a], len(views))
                M = torch.nn.functional.normalize(torch.stack([Z[torch.from_numpy(Y == c)].mean(0) for c in range(K)]), dim=1)
                oof[b] = np.mean([torch.softmax(torch.nn.functional.normalize(z(X[v][b]), dim=1) @ M.T * kw.get("T", 30), 1).numpy() for v in ev], 0)
            elif head == "lda":
                m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=kw.get("shrink", "auto"), priors=np.full(K, 1 / K))
                m.fit(np.concatenate([z(X[v][a]).numpy() for v in views]), np.tile(y[a], len(views)))
                oof[b] = np.mean([m.predict_proba(z(X[v][b]).numpy()) for v in ev], 0)
        f1, acc = G.sc(oof); np.save(f"{G.OUT}/oof/{exp}__{gn}.npy", oof); out[gn] = f1
        row = dict(exp=exp, groups=gn, f1=round(f1, 5), acc=round(acc, 5), sec=round(time.time() - t0), cfg=json.dumps(dict(names=names, views=views, part=part, norm=norm, head=head, **kw)))
        pd.DataFrame([row]).to_csv(f"{G.OUT}/results.csv", mode="a", index=False, header=not os.path.exists(f"{G.OUT}/results.csv"))
        print(f"{exp:28s} {gn:6s} F1 {f1:.4f} acc {acc:.4f} ({row['sec']}s)", flush=True)
    return out

EXPS = {
    "vithp":            dict(names=["vithp"]),
    "radio":            dict(names=["radio"]),
    "pe":               dict(names=["pe"], groups=("sig8", "sig30")),
    "siglip2":          dict(names=["siglip2"], groups=("pe8",)),
    "vithp_wd1e-4":     dict(names=["vithp"], wd=1e-4),
    "vithp_wd1":        dict(names=["vithp"], wd=1.0),
    "vithp_wd10":       dict(names=["vithp"], wd=10.0),
    "vithp_l2":         dict(names=["vithp"], norm="l2"),
    "vithp_cls":        dict(names=["vithp"], part="cls"),
    "vithp_avg":        dict(names=["vithp"], part="avg"),
    "vithp_4views":     dict(names=["vithp"], views=("plain", "hflip", "dark", "sat")),
    "vithp_nocw":       dict(names=["vithp"], cw=False),
    "vithp_ncm":        dict(names=["vithp"], head="ncm"),
    "vithp_lda":        dict(names=["vithp"], head="lda"),
    "radio_lda":        dict(names=["radio"], head="lda"),
    "radio+vithp":      dict(names=["radio", "vithp"]),
    "radio+vithp_lda":  dict(names=["radio", "vithp"], head="lda"),
    "dinov2g":          dict(names=["dinov2g"]),
    "radio3":           dict(names=["radio3"]),
    "dinov2g_lda":      dict(names=["dinov2g"], head="lda"),
    "radio3_lda":       dict(names=["radio3"], head="lda"),
    "vithp_7views":     dict(names=["vithp"], views=("plain", "hflip", "dark", "sat", "dark2", "sat2", "darksat")),
    "radio_4views":     dict(names=["radio"], views=("plain", "hflip", "dark", "sat")),
    "radio3g":          dict(names=["radio3g"]),
    "webssl1b":         dict(names=["webssl1b"]),
    "aimv2h":           dict(names=["aimv2h"]),
    "radio_7views":     dict(names=["radio"], views=("plain", "hflip", "dark", "sat", "dark2", "sat2", "darksat")),
}
if __name__ == "__main__":
    for e in (sys.argv[1:] or [k for k in EXPS if "dinov2g" not in k and "radio3" not in k]):
        try: run(e, **EXPS[e])
        except Exception as ex: print(e, "FAILED", type(ex).__name__, str(ex)[:200], flush=True)
