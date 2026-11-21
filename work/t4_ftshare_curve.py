# FT vs frozen share on true held-out scenes (SGKF folds 0-2, 17,038 rows): FT = ConvNeXt-S DINOv3 fine-tuned on the
# scene folds (the only FT run with scene-CV OOF), frozen = 4 heads equal + ViT-7B with share q. Geo-mean blend.
# Proxy only: the real FT half is 9 stronger models, so the true optimum FT share is likely a bit higher than here.
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values
lg = lambda p: np.log(np.clip(p, 1e-9, 1)); S = "t4/research/groupcv/oof_sgkf"
ft = lg(np.mean([np.load(f"t4/preds/scenecv_s/preds/v4g_cnxt_s_d3_base_oof_{v}.npy") for v in ("plain", "hflip", "up")], 0))
o4 = np.mean([lg(np.load(f"{S}/{k}.npy")) for k in ("vithp", "radio", "radio3", "dinov2g")], 0); o7 = lg(np.load(f"{S}/vit7b.npy"))
m = ~np.isnan(o7[:, 0]) & ~np.isnan(ft[:, 0]) & (np.abs(ft).sum(1) > 0)
f = lambda L: f1_score(y[m], L[m].argmax(1), average="macro")
print(f"rows {m.sum()} | FT alone {f(ft):.4f} | frozen4 {f(o4):.4f} | 7B {f(o7):.4f}")
for q in (0, .4, .7):
    fr = (1 - q) * o4 + q * o7
    print(f"7B share {q:.1f}: FT share ->", " ".join(f"{s:.2f}:{f(s * ft + (1 - s) * fr):.4f}" for s in (0, .2, .3, .4, .5, .6, .7, 1)))
