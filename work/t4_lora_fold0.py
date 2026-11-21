# LoRA 7B (val0 run) vs current clean blends on SGKF scene fold 0 (train labels only). FT proxy = scene-CV ConvNeXt-S.
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values; lg = lambda p: np.log(np.clip(p, 1e-9, 1))
D = "t4/preds/v4f_lora7b/preds"; S = "t4/research/groupcv/oof_sgkf"
ids = pd.read_csv(f"{D}/test_ids.csv")["id_img"].values; ref = pd.read_csv("t4/preds/v3_5f/preds/task4_v3_5f.csv")["id_img"].values
print("test order == old order:", (ids == ref).all(), len(ids))
va = np.load(f"{D}/lora7b_l75_val0_val_idx.npy"); yv = y[va]
lo = np.mean([lg(np.load(f"{D}/lora7b_l75_val0_oof_{v}.npy")) for v in ("plain", "hflip")], 0)
ft = lg(np.mean([np.load(f"t4/preds/scenecv_s/preds/v4g_cnxt_s_d3_base_oof_{v}.npy") for v in ("plain", "hflip", "up")], 0))[va]
o4 = np.mean([lg(np.load(f"{S}/{k}.npy")) for k in ("vithp", "radio", "radio3", "dinov2g")], 0)[va]
l75 = lg(np.load(f"{S}/vit7b_l75.npy"))[va]; s10 = lg(np.load(f"{S}/vit7b.npy"))[va]
f = lambda L: round(f1_score(yv, L.argmax(1), average="macro"), 5)
fr78 = .15 * o4 + .85 * l75; v78 = .5 * ft + .5 * fr78; v74 = .5 * ft + .5 * (.3 * o4 + .7 * s10)
print("single: lora", f(lo), "| l75", f(l75), "| s10", f(s10), "| o4", f(o4), "| cnxtS", f(ft))
print("blends: v74b", f(v74), "| v78a", f(v78))
print("a*lora + (1-a)*v78a:", {a: f(a * lo + (1 - a) * v78) for a in (0, .1, .2, .3, .4, .5, .6, .7, .8, 1)})
print("lora as FT slot (FT half = cnxtS+lora equal) :", f(.25 * ft + .25 * lo + .5 * fr78))
print("lora replaces l75 in frozen :", f(.5 * ft + .5 * (.15 * o4 + .85 * lo)))
for b in (.2, .3, .4, .5): print(f"FT .5 (cnxtS share {1-b:.1f}, lora {b:.1f}) + frozen78:", f(.5 * ((1 - b) * ft + b * lo) + .5 * fr78))
