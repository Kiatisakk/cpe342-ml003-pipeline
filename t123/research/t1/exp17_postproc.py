"""Post-processing on the final OOF (best_oof_allseeds.npy, 3 fold-split seeds). Everything nested:
every tuned quantity (threshold, group thresholds, calibrator) is fitted on 4 outer folds and applied to the 5th.

A  single positive-rate cut (reference = what final_v2 does)
A2 single probability cut
B  hard overrides in observed class-exclusive zones (no tuning)
C  per-group thresholds (groups = missingness of the 4 key features), jointly tuned by coordinate ascent on F2-macro
D  per-group isotonic calibration, then one probability threshold
E  average of the 3 split-seed OOFs (= the ensemble actually used for test), then A
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from sklearn.isotonic import IsotonicRegression

lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
O = np.load(os.path.join(ROOT, "best_oof_allseeds.npy"))
KEY = ["reports_received", "friend_network_size", "account_age_days", "crosshair_placement"]
nkey = lab[KEY].isna().sum(1).clip(0, 2).values          # group 0 / 1 / 2+ key features missing
grp_sets = {"nkey012": nkey, "rep_or_fr_missing": (lab["reports_received"].isna() | lab["friend_network_size"].isna()).astype(int).values}

z = fe_zone_post(lab, pd.concat([lab, unl]))
rep, ch, fr = lab["reports_received"], lab["crosshair_placement"], lab["friend_network_size"]
lv, ag = lab["level"], lab["account_age_days"]
force1 = ((rep == 4) | (ch < 0.82) | (lv < 28) | ((ag >= 110) & (ag < 120))).values
force0 = ((rep >= 20) | (ch > 0.94) | (lv > 96) | (ag < 15) | ((ag >= 180) & (ag < 200)) | (ag >= 1000) | (fr < 10) | (fr > 269)).values


def rate_cut(p_tr, y_tr, p_te):
    _, _, r = best_thr(y_tr, p_tr); return (p_te >= np.quantile(p_tr, 1 - r)).astype(int)


def prob_cut(p_tr, y_tr, p_te):
    t, _, _ = best_thr(y_tr, p_tr); return (p_te >= t).astype(int)


def group_thr_fit(p, yy, g, n_iter=3):
    G = np.unique(g); t0, _, _ = best_thr(yy, p); th = {k: t0 for k in G}
    for _ in range(n_iter):
        for k in G:
            m = g == k; cand = np.quantile(p[m], np.linspace(0.2, 0.9, 71))
            best = (-1, th[k])
            for c in cand:
                th2 = dict(th); th2[k] = c
                pred = (p >= np.vectorize(th2.get)(g)).astype(int); s = f2m(yy, pred)
                if s > best[0]: best = (s, c)
            th[k] = best[1]
    return th


def evaluate(name, fn, oofs, seeds=SEEDS):
    sc, sb = [], []
    for si, s in enumerate(seeds):
        p = oofs[si]; pred = np.zeros(len(y), int)
        for a, b in folds_for(y, s):
            pred[b] = fn(p, a, b)
        sc.append(f2m(y, pred)); sb.append(f2b(y, pred))
    r = dict(name=name, f2m_nested=round(np.mean(sc), 5), f2m_nested_sd=round(np.std(sc, ddof=1), 5), per_seed=[round(v, 5) for v in sc])
    print(json.dumps(r), flush=True)
    with open(os.path.join(OUT, "results_post.jsonl"), "a") as f: f.write(json.dumps(r) + "\n")
    return r


print("zone rows: force1", force1.sum(), "true-cheater rate", y[force1].mean().round(4), "| force0", force0.sum(), "rate", y[force0].mean().round(4))
pred_ref = (O[0] >= np.quantile(O[0], 1 - best_thr(y, O[0])[2])).astype(int)
print("model already predicts 1 on force1 rows:", pred_ref[force1].mean().round(4), "| 0 on force0 rows:", 1 - pred_ref[force0].mean().round(4))

evaluate("A  single rate cut (final_v2)", lambda p, a, b: rate_cut(p[a], y[a], p[b]), O)
evaluate("A2 single probability cut", lambda p, a, b: prob_cut(p[a], y[a], p[b]), O)


def zone_override(p, a, b):
    q = p.copy(); q[force1] = 1.0; q[force0] = 0.0
    return rate_cut(q[a], y[a], q[b])


evaluate("B  hard zone overrides + rate cut", zone_override, O)
for gname, g in grp_sets.items():
    def gt(p, a, b, g=g):
        th = group_thr_fit(p[a], y[a], g[a]); return (p[b] >= np.vectorize(th.get)(g[b])).astype(int)
    evaluate(f"C  per-group thresholds [{gname}]", gt, O)

    def iso(p, a, b, g=g):
        q = p.copy()
        for k in np.unique(g):
            ma = a[g[a] == k]; ir = IsotonicRegression(out_of_bounds="clip").fit(p[ma], y[ma])
            q[g == k] = ir.predict(p[g == k])
        return prob_cut(q[a], y[a], q[b])
    evaluate(f"D  per-group isotonic + one prob cut [{gname}]", iso, O)

avg = np.repeat(O.mean(0)[None, :], len(SEEDS), 0)
evaluate("E  3-seed OOF average + rate cut", lambda p, a, b: rate_cut(p[a], y[a], p[b]), avg)
