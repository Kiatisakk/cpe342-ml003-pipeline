# (1) Does random-CV OOF predict the public LB? Re-score every submitted T4 recipe (only T4 changed between these files)
#     on the same rows, and compare with its LB score.  (2) Subset search over the current model pool on OOF.
# Rows: the 21,031 train rows covered by every run (EVA02-B and the 2-fold Base cover folds 0-1 of the shared 3-fold split).
import itertools, os, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values
V = ["plain", "hflip", "up"]; lg = lambda p: np.log(np.clip(p, 1e-9, 1))
def O(d, t, vs): return np.mean([np.load(f"{d}/{t}_oof_{v}.npy") for v in vs], 0) if vs else np.load(f"{d}/{t}_oof.npy")
P = {"v3": O("t4/preds/v3_5f/preds", "v3_5f", []), "eva02s": O("t4/preds/v4_eva02_friend/preds", "v4_eva02_s", V),
     "effv2s": O("t4/preds/v4_effv2s/preds", "v4_effv2_s", V), "coloraug": O("t4/preds/v4_eva02s_coloraug/preds", "v4_eva02_s_coloraug", V),
     "cnxtS": O("t4/preds/v4_cnxt_s_d3/preds", "v4_cnxt_s_d3", V), "cnxtB2f": O("t4/preds/v4_cnxt_b_d3_f01/preds", "v4_cnxt_b_d3", V),
     "cnxtB": O("t4/preds/v4_cnxt_b_d3/preds", "v4_cnxt_b_d3", V), "cnxtL": O("t4/preds/v4_cnxt_l_d3/preds", "v4_cnxt_l_d3", V),
     "cnxtLc": O("t4/preds/v4_cnxt_l_d3_colab/preds", "v4_cnxt_l_d3c", V), "eva02b": O("t4/preds/v4_eva02_b/preds", "v4_eva02_b", V),
     "vithp": O("t4/preds/dinov3_head/vithp", "d3lin_vithp", []), "radio": O("t4/preds/dinov3_head/radio", "d3lin_radio", []),
     "radio3": O("t4/preds/dinov3_head/radio3", "d3lin_radio3", []), "dinov2g": O("t4/preds/dinov3_head/dinov2g", "d3lin_dinov2g", []),
     "vithp_ph": O("t4/preds/dinov3_head/vithp_ph", "d3lin_vithp_ph", []), "radio_ph": O("t4/preds/dinov3_head/radio_ph", "d3lin_radio_ph", []),
     "radio3_ph": O("t4/preds/dinov3_head/radio3_ph", "d3lin_radio3_ph", []), "dinov2g_ph": O("t4/preds/dinov3_head/dinov2g_ph", "d3lin_dinov2g_ph", [])}
m = np.all([~np.isnan(p[:, 0]) for p in P.values()], 0)
import sys
if "--hard" in sys.argv:      # "hard" rows: bottom 25% of train->train nearest-neighbour cosine (RADIOv3, train only) = images
    nn1 = np.load("t4/research/groupcv/train_nn1_radio3.npy")   # with no near-duplicate in train, like new scenes
    m &= nn1 <= np.quantile(nn1, .25); print("HARD rows only (no close duplicate in train)")
ym = y[m]; L = {k: lg(v[m]) for k, v in P.items()}
print("common rows", m.sum())
f1 = lambda z: f1_score(ym, z.argmax(1), average="macro")
def bl(w): return f1(sum(v * L[k] for k, v in w.items()) / sum(w.values()))

# (1) submitted recipes with no EM (T4 is the only difference between them). LB from CLAUDE.md.
S = [("v25a  v3 .6 + eva .4", {"v3": .6, "eva02s": .4}, 0.85787),
     ("v26   eq3 + effv2s", {"v3": 1, "eva02s": 1, "effv2s": 1}, 0.85630),
     ("v26c  .5/.3/.2 coloraug", {"v3": .5, "eva02s": .3, "coloraug": .2}, 0.85733),
     ("v28   .4/.4/.2 cnxtS", {"v3": .4, "eva02s": .4, "cnxtS": .2}, 0.85906),
     ("v30   eq4 (B 2f)", {"v3": 1, "eva02s": 1, "cnxtS": 1, "cnxtB2f": 1}, 0.86169),
     ("v31   FT .7 + frozen .3", {"v3": .175, "eva02s": .175, "cnxtS": .175, "cnxtB2f": .175, "radio": .15, "vithp": .15}, 0.86340),
     ("v32   FT .5 + frozen .5", {"v3": .125, "eva02s": .125, "cnxtS": .125, "cnxtB2f": .125, "radio": .25, "vithp": .25}, 0.86430),
     ("v37n  FT5 + frozen-4", {"v3": .1, "eva02s": .1, "cnxtS": .1, "cnxtB": .1, "cnxtL": .1, "vithp": .2, "radio": .05, "radio3": .1, "dinov2g": .15}, 0.86494)]
print(f"\n{'recipe':28s} {'OOF':>8s} {'LB':>8s}")
oo, lb = [], []
for n, w, s in S:
    o = bl(w); oo.append(o); lb.append(s); print(f"{n:28s} {o:.5f} {s:.5f}")
r = spearmanr(oo, lb); print(f"Spearman OOF vs LB over {len(S)} files: {r.correlation:.2f}")
for i in range(1, len(S)):
    print(f"  {S[i-1][0][:5]} -> {S[i][0][:5]}: OOF {'up' if oo[i] > oo[i-1] else 'down':4s} {oo[i]-oo[i-1]:+.5f}  LB {'up' if lb[i] > lb[i-1] else 'down':4s} {lb[i]-lb[i-1]:+.5f}")

# (2) subset search: FT pool x frozen (clean _ph heads), FT group .5 / frozen group .5, equal inside each group
FTP = ["v3", "eva02s", "cnxtS", "cnxtB", "cnxtL", "cnxtLc", "eva02b"]; FRP = ["vithp_ph", "radio_ph", "radio3_ph", "dinov2g_ph"]
res = []
for a in range(1, len(FTP) + 1):
    for ft in itertools.combinations(FTP, a):
        for b in range(0, len(FRP) + 1):
            for fr in itertools.combinations(FRP, b):
                w = {k: .5 / len(ft) if fr else 1 / len(ft) for k in ft}; w.update({k: .5 / len(fr) for k in fr})
                res.append((bl(w), ft, fr))
res.sort(key=lambda r: -r[0])
print(f"\nsubsets scored: {len(res)}; top 10 by OOF:")
for s, ft, fr in res[:10]: print(f"  {s:.5f}  FT {'+'.join(ft):45s} frozen {'+'.join(fr) or '-'}")
ref = {"v43 (FT5 + frozen4)": (tuple(FTP[:5]), tuple(FRP)), "v48 (+eva02b)": (tuple(FTP[:5] + ['eva02b']), tuple(FRP)),
       "v49 (FT7 + frozen4)": (tuple(FTP), tuple(FRP)), "FT7 only": (tuple(FTP), ())}
rank = {(ft, fr): i for i, (_, ft, fr) in enumerate(res)}
for n, k in ref.items(): print(f"  {n:22s} OOF {res[rank[k]][0]:.5f}  rank {rank[k] + 1}/{len(res)}")
best_by_nfr = {}
for s, ft, fr in res: best_by_nfr.setdefault(len(fr), (s, ft, fr))
print("best with n frozen heads:", {k: (round(v[0], 5), "+".join(v[1])) for k, v in sorted(best_by_nfr.items())})
