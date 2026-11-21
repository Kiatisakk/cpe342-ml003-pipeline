# Frozen-only blend on scene-group OOF (sig8 / sig30): share of the ViT-7B head inside the frozen group, the other
# 4 heads equal. Train labels only. Tells whether "more weight on the strong head" keeps helping on unseen scenes.
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values
lg = lambda p: np.log(np.clip(p, 1e-9, 1)); O = "t4/research/groupcv/oof"
for g in ("sig8", "sig30"):
    o4 = np.mean([lg(np.load(f"{O}/{k}__{g}.npy")) for k in ("vithp", "radio", "radio3", "dinov2g")], 0)
    o7 = lg(np.load(f"{O}/vit_7b_patch16_dinov3_phg__{g}.npy"))
    row = [f"{w:.1f}:{f1_score(y, ((1 - w) * o4 + w * o7).argmax(1), average='macro'):.4f}" for w in (0, .2, .4, .5, .6, .7, .8, .9, 1)]
    print(g, "7B share of frozen ->", " ".join(row))
