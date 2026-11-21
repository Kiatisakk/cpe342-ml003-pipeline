# Local step after the Kaggle kernel task4-frozen-feats-ms (multi-scale frozen features + heads, GPU).
#   python -m kaggle kernels output kiatisakkk/task4-frozen-feats-ms -p t4/preds/frozen_ms
#   python work/t4_ms_pipeline.py t4/preds/frozen_ms/heads
# 1. imports every head into t4/preds/dinov3_head/<name>/ (test rows re-ordered to the OLD order used by every other run)
# 2. prints random-CV OOF and hard-row macro-F1 of each head next to the existing single-scale clean head (<bk>_ph)
# 3. builds two pre-registered clean T4 blends on the v64a model set (FT .5 equal: v3, Base full, Large full x2, EVA02-B,
#    Large v3-aug; frozen .5 equal over 4 backbones), swapping each backbone's _ph head for its multi-scale head:
#      ms  = early fusion of scales s05 + s10 + s15        -> sub_v67a_clean_ms.csv
#      msl = ms + intermediate blocks l50 + l75            -> sub_v67b_clean_msl.csv
#    A backbone without a finished multi-scale head keeps its _ph head. Nothing is submitted.
import sys, os, subprocess, numpy as np, pandas as pd
from sklearn.metrics import f1_score
H = sys.argv[1]
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique())
y = tr["class"].map({c: i for i, c in enumerate(C)}).values
ref = pd.read_csv("t4/preds/v3_5f/preds/task4_v3_5f.csv")["id_img"].astype(str).values          # OLD test order
nn1 = np.load("t4/research/groupcv/train_nn1_radio3.npy"); hard = nn1 <= np.quantile(nn1, .25)
D = "t4/preds/dinov3_head"; BKS = ["vithp", "radio", "radio3", "dinov2g"]                        # the clean frozen set

trn = np.load(f"{H}/train_names.npy", allow_pickle=True); ten = np.load(f"{H}/test_names.npy", allow_pickle=True)
assert (trn == tr["id"].astype(str).str.replace(".jpg", "", regex=False).values).all(), "train order differs from the labels"
pos = pd.Series(np.arange(len(ten)), index=ten).loc[ref].values
names = sorted(n for n in os.listdir(H) if os.path.isfile(f"{H}/{n}/d3lin_{n}_oof.npy"))
print("heads found:", names)
sc = lambda o: (f1_score(y, o.argmax(1), average="macro"), f1_score(y[hard], o[hard].argmax(1), average="macro"))
rows = []
for n in names:
    od = f"{D}/{n}"; os.makedirs(od, exist_ok=True); oof = np.load(f"{H}/{n}/d3lin_{n}_oof.npy"); np.save(f"{od}/d3lin_{n}_oof.npy", oof)
    for k in range(5): np.save(f"{od}/d3lin_{n}_test_fold{k}.npy", np.load(f"{H}/{n}/d3lin_{n}_test_fold{k}.npy")[pos])
    rows.append((n, *sc(oof)))
for bk in BKS + ["radio3g"]:
    if os.path.isfile(f"{D}/{bk}_ph/d3lin_{bk}_ph_oof.npy") and any(n.startswith(bk + "_") for n in names):
        rows.append((f"{bk}_ph (existing, single scale)", *sc(np.load(f"{D}/{bk}_ph/d3lin_{bk}_ph_oof.npy"))))
print(pd.DataFrame(rows, columns=["head", "OOF macroF1", "hard rows"]).sort_values("head").round(5).to_string(index=False))

FT = "v3,cnxtBf,cnxtLf+cnxtLfk,eva02b,cnxtLv3"                                                 # = sub_v64a_clean_Lv3
for var, out in (("ms", "sub_v67a_clean_ms.csv"), ("msl", "sub_v67b_clean_msl.csv")):
    fr = [(f"{bk}_{var}" if f"{bk}_{var}" in names else f"{bk}_ph") for bk in BKS]
    if all(f.endswith("_ph") for f in fr): print(f"\n{var}: no multi-scale head finished -> skipped"); continue
    print(f"\n== {var}: frozen heads {fr}")
    env = dict(os.environ, T4_FR=",".join(f"{f}:.125" for f in fr))
    subprocess.run([sys.executable, "work/t4_build_subset.py", f"clean_{var}", FT, "clean"], check=True, env=env)
    subprocess.run([sys.executable, "submissions/tools/build_clean.py", f"t4/research/blend_clean_{var}", out], check=True)
    a = pd.read_csv(f"submissions/FINAL/{out}", dtype=str); b = pd.read_csv("submissions/FINAL/sub_v64a_clean_Lv3.csv", dtype=str)
    print(out, "vs sub_v64a_clean_Lv3:", {c: int((a[c] != b[c]).sum()) for c in a.columns[1:] if (a[c] != b[c]).any()})
print("\nnothing submitted")
