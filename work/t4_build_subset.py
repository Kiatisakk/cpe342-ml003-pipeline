# Build a T4 blend from a chosen fine-tuned subset. FT group .5 (equal), frozen group .5.
# usage: python work/t4_build_subset.py <out name> <ft keys comma> clean|v37em
#   clean = frozen heads plain+hflip (_ph), equal weights, no EM;  v37em = v37 frozen heads/weights + EM prior (NOT clean)
import sys, os, numpy as np, pandas as pd
from sklearn.metrics import f1_score
name, fts, mode = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique()); pt = tr["class"].value_counts(normalize=True).reindex(C).values
y = tr["class"].map({c: i for i, c in enumerate(C)}).values
V = ["plain", "hflip", "up"]; lg = lambda p: np.log(np.clip(p, 1e-9, 1))
FT = {"v3": ("t4/preds/v3_5f/preds", "v3_5f", []), "eva02s": ("t4/preds/v4_eva02_friend/preds", "v4_eva02_s", V), "cnxtS": ("t4/preds/v4_cnxt_s_d3/preds", "v4_cnxt_s_d3", V),
      "cnxtB": ("t4/preds/v4_cnxt_b_d3/preds", "v4_cnxt_b_d3", V), "cnxtL": ("t4/preds/v4_cnxt_l_d3/preds", "v4_cnxt_l_d3", V),
      "cnxtLc": ("t4/preds/v4_cnxt_l_d3_colab/preds", "v4_cnxt_l_d3c", V), "eva02b": ("t4/preds/v4_eva02_b/preds", "v4_eva02_b", V),
      "cnxtLf": ("t4/preds/v4f_cnxt_l_d3c", "v4f_cnxt_l_d3c_s42", V),            # full-train refit (no folds, no OOF), Colab A100
      "cnxtLfk": ("t4/preds/v4f_cnxt_l_d3c_kg/preds", "v4f_cnxt_l_d3c_s42", V),  # full-train Large, Kaggle T4 (batch 16 x accum 2)
      "cnxtBf": ("t4/preds/v4f_cnxt_b_d3_nolq/preds", "v4f_cnxt_b_d3_s42_nolq", V),  # full-train Base, JPEG-like aug OFF
      "dinov2l": ("t4/preds/v4_dinov2_l/preds", "v4_dinov2_l", V),
      "cnxtLv3": ("t4/preds/v4f_cnxt_l_d3c_v3aug", "v4f_cnxt_l_d3c_v3aug_s42", V),
      "cnxtLv3k": ("t4/preds/v4f_cnxt_l_d3_v3aug_kg/preds", "v4f_cnxt_l_d3c_v3aug_s42", V),  # full-train Large, v3 aug, Kaggle T4
      "eva02Lf": ("t4/preds/v4f_eva02_l", "v4f_eva02_l_v4aug_s42", V),            # full-train EVA02-L (v4 recipe, Colab A100)
      "vitLf": ("t4/preds/v4f_vit_l_d3", "v4f_vit_l_d3_v4aug_s42", V)}            # full-train DINOv3 ViT-L/16 (v4 recipe, Colab A100)           # full-train EVA02-L (v4 recipe, Colab A100)  # full-train Large, v3 augmentation (Colab)               # DINOv2 ViT-L fine-tune, fold 0 only
# an FT key "a+b" = one slot holding the geometric mean of a and b
FR = {"clean": [(f"t4/preds/dinov3_head/{s}_ph", f"d3lin_{s}_ph", .125) for s in ("vithp", "radio", "radio3", "dinov2g")],
      "v37em": [("t4/preds/dinov3_head/vithp", "d3lin_vithp", .20), ("t4/preds/dinov3_head/radio", "d3lin_radio", .05),
                ("t4/preds/dinov3_head/radio3", "d3lin_radio3", .10), ("t4/preds/dinov3_head/dinov2g", "d3lin_dinov2g", .15)]}[mode]
if os.environ.get("T4_FR"):            # override frozen heads, e.g. "vithp:.2,radio:.05,radio3g:.1,dinov2g:.15" (weights sum to .5)
    FR = [(f"t4/preds/dinov3_head/{n}", f"d3lin_{n}", float(x)) for n, x in (p.split(":") for p in os.environ["T4_FR"].split(","))]
def T(d, t, vs):
    if os.path.exists(f"{d}/{t}_test_full_{vs[0] if vs else 'plain'}.npy"): return np.mean([np.load(f"{d}/{t}_test_full_{v}.npy") for v in vs], 0)
    o = []
    for k in range(5):
        fs = [f"{d}/{t}_test_fold{k}_{v}.npy" for v in vs] if vs else [f"{d}/{t}_test_fold{k}.npy"]
        if os.path.exists(fs[0]): o.append(np.mean([np.load(x) for x in fs], 0))
    return np.mean(o, 0)
def O(d, t, vs):
    if os.path.exists(f"{d}/{t}_test_full_{vs[0] if vs else 'plain'}.npy"): return None          # full-train run: no OOF
    return np.mean([np.load(f"{d}/{t}_oof_{v}.npy") for v in vs], 0) if vs else np.load(f"{d}/{t}_oof.npy")
def slot(k, fn):                                                       # log-prob of one FT slot ("a+b" = geo-mean)
    parts = [fn(*FT[p]) for p in k.split("+")]; parts = [p for p in parts if p is not None]
    return np.mean([lg(p) for p in parts], 0) if parts else None
SH = float(os.environ.get("T4_FROZEN_SHARE", ".5"))                # frozen group share (LB-chosen .5)
WM = float(os.environ.get("T4_MLP", "0"))                      # concat-feature MLP head weight, taken from the frozen share
w = [(k, (1 - SH) / len(fts)) for k in fts] + [((d, t, []), a / .5 * (SH - WM)) for d, t, a in FR]
if WM > 0: w.append((("t4/preds/dinov3_head/concat4_mlp", "d3lin_concat4_mlp", []), WM))
Lt = sum(a * (slot(s, T) if isinstance(s, str) else lg(T(*s))) for s, a in w); P = np.exp(Lt - Lt.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
lo = [(slot(s, O) if isinstance(s, str) else lg(O(*s)), a) for s, a in w]
if any(z is None for z, _ in lo): print("note: a slot has no OOF (full-train run) -> OOF below excludes it")
lo = [(z, a) for z, a in lo if z is not None]; Lo = sum(a * z for z, a in lo) / sum(a for _, a in lo); m = ~np.isnan(Lo[:, 0]); nn1 = np.load("t4/research/groupcv/train_nn1_radio3.npy"); h = m & (nn1 <= np.quantile(nn1, .25))
print(f"{name}: OOF rows {m.sum()} macroF1 {f1_score(y[m], Lo[m].argmax(1), average='macro'):.5f} | hard rows {h.sum()} {f1_score(y[h], Lo[h].argmax(1), average='macro'):.5f}")
if mode == "v37em":
    pi = pt.copy()
    for _ in range(300):
        Q = P * pi / pt; Q /= Q.sum(1, keepdims=True); n = Q.mean(0)
        if np.abs(n - pi).max() < 1e-7: break
        pi = n
    P = P * pi / pt; P /= P.sum(1, keepdims=True)
od = f"t4/research/blend_{name}"; os.makedirs(od, exist_ok=True); ref = pd.read_csv("t4/preds/v3_5f/preds/task4_v3_5f.csv")
np.save(f"{od}/t4_blend_test_proba.npy", P.astype(np.float32))
pd.DataFrame({"id": ref["id"], "id_img": ref["id_img"], "pred_num": P.argmax(1), "pred_str": np.array(C)[P.argmax(1)]}).to_csv(f"{od}/t4_blend.csv", index=False, lineterminator="\n")
print("->", od)
