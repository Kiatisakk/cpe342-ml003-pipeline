# EVA02-L full-train has no OOF. Label-free test checks (no test labels, nothing tuned on test):
# agreement of each FT run with (a) the consensus of the OTHER FT runs, (b) the frozen heads (different training = an
# independent opinion), incl. the new ViT-7B head; split by how hard the image is (low consensus confidence).
import sys, numpy as np
sys.argv = ["x", "n", "v3", "clean"]
exec(open("work/t4_build_subset.py", encoding="utf-8").read().split("SH = float")[0])   # FT, T, lg
keys = ["v3", "cnxtBf", "cnxtLf", "cnxtLfk", "eva02b", "cnxtLv3", "cnxtLv3k", "eva02Lf", "vitLf"]
P = {k: T(*FT[k]) for k in keys}
fr = {s: np.mean([np.load(f"t4/preds/dinov3_head/{s}/d3lin_{s}_test_fold{i}.npy") for i in range(5)], 0)
      for s in ("vithp_ph", "radio_ph", "radio3_ph", "dinov2g_ph", "vit_7b_patch16_dinov3_ph")}
F4 = np.mean([lg(fr[s]) for s in list(fr)[:4]], 0).argmax(1); F7 = fr["vit_7b_patch16_dinov3_ph"].argmax(1)
allc = np.exp(np.mean([lg(P[k]) for k in keys], 0)); conf = allc.max(1) / allc.sum(1); hard = conf < np.quantile(conf, .2)
print(f"{'run':9s} {'vs other FT':>11s} {'(hard 20%)':>10s} {'vs frozen4':>10s} {'(hard)':>7s} {'vs 7B':>7s} {'(hard)':>7s}")
for k in keys:
    o = np.mean([lg(P[j]) for j in keys if j != k], 0).argmax(1); a = P[k].argmax(1)
    print(f"{k:9s} {(a == o).mean():11.3f} {(a == o)[hard].mean():10.3f} {(a == F4).mean():10.3f} {(a == F4)[hard].mean():7.3f} "
          f"{(a == F7).mean():7.3f} {(a == F7)[hard].mean():7.3f}")
