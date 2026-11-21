# Score the frozen ViT-7B head next to the 4 clean _ph heads (random OOF and hard rows), and frozen-only blends.
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values
nn1 = np.load("t4/research/groupcv/train_nn1_radio3.npy"); h = nn1 <= np.quantile(nn1, .25)
H = {s: np.load(f"t4/preds/dinov3_head/{s}/d3lin_{s}_oof.npy") for s in
     ("vithp_ph", "radio_ph", "radio3_ph", "dinov2g_ph", "vit_7b_patch16_dinov3_ph")}
lg = lambda p: np.log(np.clip(p, 1e-9, 1)); sc = lambda L: (f1_score(y, L.argmax(1), average="macro"), f1_score(y[h], L[h].argmax(1), average="macro"))
for k, v in H.items(): print(f"{k:28s} OOF {sc(v)[0]:.5f} hard {sc(v)[1]:.5f}")
B = {"frozen4 (clean)": ["vithp_ph", "radio_ph", "radio3_ph", "dinov2g_ph"],
     "frozen5 (+7B)": ["vithp_ph", "radio_ph", "radio3_ph", "dinov2g_ph", "vit_7b_patch16_dinov3_ph"],
     "7B replaces ViT-H+": ["vit_7b_patch16_dinov3_ph", "radio_ph", "radio3_ph", "dinov2g_ph"]}
for n, ks in B.items(): o, hh = sc(np.mean([lg(H[k]) for k in ks], 0)); print(f"{n:28s} OOF {o:.5f} hard {hh:.5f}")
