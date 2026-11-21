# Was the FT set chosen, or just blended? Check every FT run that has OOF: random OOF and "hard rows" (images whose
# nearest train neighbour is far, bottom 25% NN cosine on RADIOv3 train features = the closest proxy of held-out scenes
# available for FT models; scene-group OOF exists only for ConvNeXt-S). Then FT-only blends: family-best set vs all,
# and leave-one-out from the family-best set. Train labels only.
import sys, itertools, numpy as np
sys.argv = ["x", "n", "v3", "clean"]
exec(open("work/t4_build_subset.py", encoding="utf-8").read().split("SH = float")[0])   # FT, O, y, lg, f1_score
nn1 = np.load("t4/research/groupcv/train_nn1_radio3.npy"); H = nn1 <= np.quantile(nn1, .25)
keys = ["v3", "eva02s", "cnxtS", "cnxtB", "cnxtL", "cnxtLc", "eva02b", "dinov2l"]
Ob = {k: O(*FT[k]) for k in keys}
f = lambda L, m: f1_score(y[m], L[m].argmax(1), average="macro")
print(f"{'run':8s} {'rows':>6s} {'OOF':>8s} {'hard':>8s}")
for k in keys:
    m = ~np.isnan(Ob[k][:, 0]) & (Ob[k].sum(1) > 0); print(f"{k:8s} {m.sum():6d} {f(Ob[k], m):8.5f} {f(Ob[k], m & H):8.5f}")
full = [k for k in keys if k != "dinov2l"]; M = np.ones(len(y), bool)
for k in full: M &= (Ob[k].sum(1) > 0)
bl = lambda ks: np.mean([lg(Ob[k]) for k in ks], 0)
print(f"\nFT-only geo-mean blends on {M.sum()} rows (hard {(M & H).sum()})")
best = ["v3", "cnxtB", "cnxtLc", "eva02b"]
S = {"family-best (v3,B,Lcolab,EVA02-B)": best, "all 7": full, "v3+EVA02-S (v23)": ["v3", "eva02s"],
     "family-worst (v3,S,Lkaggle,EVA02-S)": ["v3", "cnxtS", "cnxtL", "eva02s"]}
for n, ks in S.items(): print(f"  {n:38s} OOF {f(bl(ks), M):.5f} hard {f(bl(ks), M & H):.5f}")
print("  leave-one-out from family-best:")
for k in best: ks = [x for x in best if x != k]; print(f"    - {k:8s} OOF {f(bl(ks), M):.5f} hard {f(bl(ks), M & H):.5f}")
R = []
for r in range(2, len(full) + 1):
    for ks in itertools.combinations(full, r): R.append((f(bl(ks), M & H), f(bl(ks), M), ks))
R.sort(reverse=True); rk = [x[2] for x in R].index(tuple(k for k in full if k in best)) + 1
print(f"\nall {len(R)} subsets ranked by hard rows: family-best is #{rk}; top 5:")
for h, o, ks in R[:5]: print(f"  hard {h:.5f} OOF {o:.5f} {ks}")
print(f"  median subset hard {np.median([x[0] for x in R]):.5f}")
