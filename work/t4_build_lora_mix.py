# T4 blend = a * LoRA-7B full-train (plain+hflip) + (1-a) * an existing blend, in log space. Optional EM prior (NOT clean).
# usage: python work/t4_build_lora_mix.py <base blend dir> <a> <out name> clean|em|trainprior  (trainprior = P x train_prior^-1, train counts only)
import sys, os, numpy as np, pandas as pd
base, a, name, mode = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv"); C = sorted(tr["class"].unique()); pt = tr["class"].value_counts(normalize=True).reindex(C).values
lg = lambda p: np.log(np.clip(p, 1e-9, 1)); D = "t4/preds/v4f_lora7b/preds"
ref = pd.read_csv("t4/preds/v3_5f/preds/task4_v3_5f.csv"); assert (pd.read_csv(f"{D}/test_ids.csv")["id_img"].values == ref["id_img"].values).all()
# env T4_LORA = "dir:tag:w[:sizes],..." (LoRA slot = weighted mean of log-probs); sizes = ";"-list of extra TTA sizes, e.g. 192x352;256x480
# (the base 224x416 files are always included, all sizes averaged equally); default = the l75 seed-42 run
spec = [x.split(":") for x in os.environ.get("T4_LORA", f"{D}:v4f_lora7b_l75_s42:1").split(",")]
for d, t, *_ in spec: assert (pd.read_csv(f"{d}/test_ids.csv")["id_img"].values == ref["id_img"].values).all() if os.path.exists(f"{d}/test_ids.csv") else True
def slot(d, t, w, sz=""):
    sx = [""] + [f"_{x}" for x in sz.split(";") if x]
    return np.mean([lg(np.load(f"{d}/{t}_test_full_{v}{x}.npy")) for v in ("plain", "hflip") for x in sx], 0)
lo = sum(float(e[2]) * slot(*e) for e in spec) / sum(float(e[2]) for e in spec)
Pb = np.load(f"{base}/t4_blend_test_proba.npy")
L = a * lo + (1 - a) * lg(Pb); P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
if mode == "em":
    pi = pt.copy()
    for _ in range(300):
        Q = P * pi / pt; Q /= Q.sum(1, keepdims=True); n = Q.mean(0)
        if np.abs(n - pi).max() < 1e-7: break
        pi = n
    P = P * pi / pt; P /= P.sum(1, keepdims=True); print("EM prior", np.round(pi, 3))
if mode == "trainprior": P = P / pt; P /= P.sum(1, keepdims=True)
od = f"t4/research/blend_{name}"; os.makedirs(od, exist_ok=True); np.save(f"{od}/t4_blend_test_proba.npy", P.astype(np.float32))
pd.DataFrame({"id": ref["id"], "id_img": ref["id_img"], "pred_num": P.argmax(1), "pred_str": np.array(C)[P.argmax(1)]}).to_csv(f"{od}/t4_blend.csv", index=False, lineterminator="\n")
print(f"{name}: agree LoRA {np.mean(P.argmax(1) == lo.argmax(1)):.4f} | rows changed vs base {(P.argmax(1) != Pb.argmax(1)).sum()} | LoRA alone vs base {(lo.argmax(1) != Pb.argmax(1)).sum()}")
