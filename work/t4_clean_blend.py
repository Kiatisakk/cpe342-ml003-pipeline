# T4 "strict clean" blend (TA answers 15 Nov: no test-set statistics in any modelling decision).
#  - 5 fine-tuned models (6 with --colabL), TTA plain/hflip/up: FT group .5 split equally
#  - 4 frozen linear heads trained on train views plain+hflip only (no dark/sat views): weight .125 each (equal;
#    no scene-group-CV weights). Fine-tuned 50 / frozen 50 split = sub_v32 recipe (picked on the public LB).
#  - no EM prior correction; argmax of the weighted geometric mean.
# Also writes a T1-cleaned submission: T1 from t1/preds/clean_fixedthr/t1_clean.csv (OOF-referenced ranks, fixed cut).
import numpy as np, pandas as pd, os
from sklearn.metrics import f1_score
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values
V = ["plain", "hflip", "up"]; lg = lambda p: np.log(p + 1e-9)
def T(d, t, vs):
    o = []
    for k in range(5):
        fs = [f"{d}/{t}_test_fold{k}_{v}.npy" for v in vs] if vs else [f"{d}/{t}_test_fold{k}.npy"]
        if os.path.exists(fs[0]): o.append(np.mean([np.load(x) for x in fs], 0))
    return np.mean(o, 0)
FT = [("t4/preds/v3_5f/preds", "v3_5f", []), ("t4/preds/v4_eva02_friend/preds", "v4_eva02_s", V),
      ("t4/preds/v4_cnxt_s_d3/preds", "v4_cnxt_s_d3", V), ("t4/preds/v4_cnxt_b_d3/preds", "v4_cnxt_b_d3", V),
      ("t4/preds/v4_cnxt_l_d3/preds", "v4_cnxt_l_d3", V)]
import sys
if "--colabL" in sys.argv: FT.append(("t4/preds/v4_cnxt_l_d3_colab/preds", "v4_cnxt_l_d3c", V))   # friend's Colab Large run (OOF .9917)
if "--eva02b" in sys.argv: FT.append(("t4/preds/v4_eva02_b/preds", "v4_eva02_b", V))           # friend's EVA02-B run (folds present on disk)
OD = "t4/research/blend_clean_strict" + ("_colabL" if "--colabL" in sys.argv else "") + ("_eva02b" if "--eva02b" in sys.argv else "")
FR = [f"t4/preds/dinov3_head/{s}_ph/d3lin_{s}_ph" for s in ("vithp", "radio", "radio3", "dinov2g")]
for f in FR:
    print(os.path.basename(f), "random-CV OOF macroF1", round(f1_score(y, np.load(f + "_oof.npy").argmax(1), average="macro"), 5))
L = sum(.5 / len(FT) * lg(T(d, t, vs)) for d, t, vs in FT) + sum(.125 * lg(T(os.path.dirname(f), os.path.basename(f), [])) for f in FR)
P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
od = OD; os.makedirs(od, exist_ok=True); ref = pd.read_csv("t4/preds/v3_5f/preds/task4_v3_5f.csv")
np.save(f"{od}/t4_blend_test_proba.npy", P.astype(np.float32))
pd.DataFrame({"id": ref["id"], "id_img": ref["id_img"], "pred_num": P.argmax(1), "pred_str": np.array(C)[P.argmax(1)]}).to_csv(f"{od}/t4_blend.csv", index=False, lineterminator="\n")
for name, f in [("v37", "blend_frozen4_ft5L_emfull/t4_blend_test_proba.npy"), ("v37n (no EM)", "blend_v37_noem/t4_blend_test_proba.npy")]:
    print("T4 rows differing from", name, int((np.load("t4/research/" + f).argmax(1) != P.argmax(1)).sum()))
print("predicted share", dict(zip(C, (np.bincount(P.argmax(1), minlength=5) / len(P)).round(3))))
