# Builds t4/colab/task4_frozen_dinov3_7b_ms_colab.ipynb: multi-scale + intermediate-layer frozen DINOv3 ViT-7B features
# (Colab A100). Complements the s10 run already in t4/preds/frozen_7b/feats:
#   s05 (112x208) and s15 (336x640) final features, l50 / l75 = blocks at 50% / 75% depth hooked during an s10 pass.
# Views plain + hflip for train and test (clean). Files feats/vit7b_{s05,s15,l50,l75}_{split}_{view}.npy, float16,
# [first token, mean patch]. Local heads (one per tag):
#   python work/t4_head_lowmem.py t4/preds/frozen_7b_ms/feats vit7b_s15 plain,hflip _ph      (same for s05, l50, l75)
import json
C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: C.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# Task 4: DINOv3 ViT-7B multi-scale + intermediate layers (Colab A100)

Features only (no training, nothing submitted). Adds to the earlier 7B run: scale s05 and s15, and blocks at 50% / 75% depth.

**Before you start**
1. Runtime → **A100 GPU** + **High-RAM**.
2. 🔑 Secrets: `HF_TOKEN` (DINOv3 licence accepted), Notebook access on.
3. `task4_data.zip` in `MyDrive/cpe342_t4/`.
4. Run all. Time ~3-3.5 h on an A100 (order: layers l50/l75 first, then s15, then s05). Every file is saved to Drive when done;
   after a disconnect **Run all** again skips finished files.

**Output**: `MyDrive/cpe342_t4/frozen_7b_ms/feats/` (~5 GB) and `MyDrive/cpe342_t4/frozen_7b_ms_feats.zip`. Send the zip back
(or the finished .npy files if the session ran out of time).
""")

code("""
# ===== 1. settings
MODEL = "vit_7b_patch16_dinov3.lvd1689m"
JOBS  = [("layers", (224, 416), 24), ("s15", (336, 640), 12), ("s05", (112, 208), 64)]   # (job, (H, W), batch); patch 16
LAYERS = {"l50": 0.5, "l75": 0.75}
VIEWS = {"test": ["plain", "hflip"], "train": ["plain", "hflip"]}
DRIVE_DIR = "/content/drive/MyDrive/cpe342_t4"
""")

code("""
# ===== 2. GPU, packages, Drive, secrets
import os, subprocess, torch, importlib.metadata as im
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> A100"
try: tv = im.version("timm")
except im.PackageNotFoundError: tv = None
if tv != "1.0.30": subprocess.run(["pip", "install", "-q", "timm==1.0.30"], check=True)
import timm, timm.data
from google.colab import drive, userdata
drive.mount("/content/drive")
class SetupError(Exception): pass
try: os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception as e: raise SetupError(f"add Colab secret HF_TOKEN and turn on Notebook access [{type(e).__name__}]") from None
OUT = f"{DRIVE_DIR}/frozen_7b_ms/feats"; os.makedirs(OUT, exist_ok=True)
print("timm", timm.__version__, "| out", OUT)
""")

code("""
# ===== 3. data -> local SSD, uint8 arrays
import glob, zipfile, time, numpy as np, pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
DATA = "/content/data"; os.makedirs(DATA, exist_ok=True)
if not glob.glob(f"{DATA}/**/train.csv", recursive=True):
    z = f"{DRIVE_DIR}/task4_data.zip"
    if not os.path.exists(z): raise SetupError(f"upload task4_data.zip to {z}")
    zipfile.ZipFile(z).extractall(DATA)
ROOT = [os.path.dirname(p) for p in glob.glob(f"{DATA}/**/train.csv", recursive=True)][0]
train = pd.read_csv(f"{ROOT}/train.csv"); test = pd.read_csv(f"{ROOT}/test.csv")
tr_names = train["id"].astype(str).values; te_names = test["id"].astype(str).values
np.save(f"{OUT}/train_names.npy", tr_names); np.save(f"{OUT}/test_names.npy", te_names)
def load_all(folder, names):   # cache 448x240 (~19 GB uint8 for train+test, High-RAM) so s15 is not upsampled from 230x120
    read = lambda i: np.asarray(Image.open(f"{folder}/{i}.jpg").convert("RGB").resize((448, 240)))
    with ThreadPoolExecutor(32) as ex: return torch.from_numpy(np.stack(list(ex.map(read, names, chunksize=64))))
t0 = time.time(); X = {"train": load_all(f"{ROOT}/train", tr_names), "test": load_all(f"{ROOT}/test", te_names)}
print({k: tuple(v.shape) for k, v in X.items()}, f"{time.time()-t0:.0f}s")
""")

code("""
# ===== 4. model (bf16) + transformer blocks for the layer hooks
import torch.nn.functional as F, gc
t0 = time.time()
model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=(224, 416), dynamic_img_size=True)
cfg = timm.data.resolve_model_data_config(model)
model = model.to(torch.bfloat16).to("cuda").eval(); gc.collect()
MEAN = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1); STD = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
NP = model.num_prefix_tokens; BLOCKS = model.blocks
LIDX = {k: int(round(f * len(BLOCKS))) - 1 for k, f in LAYERS.items()}
print(f"{sum(p.numel() for p in model.parameters())/1e9:.1f}B params | {len(BLOCKS)} blocks, hooks {LIDX} | prefix {NP} | {time.time()-t0:.0f}s")
""")

code("""
# ===== 5. extract; each file saved to Drive (re-running skips finished files)
import json
pool = lambda t: torch.cat([t[:, 0], t[:, NP:].mean(1)], 1).float().cpu().numpy().astype(np.float16)
@torch.no_grad()
def extract(Xs, v, size, bs, layers):
    store, hooks, outs = {}, [], {}
    if layers:
        for k, i in LIDX.items(): hooks.append(BLOCKS[i].register_forward_hook(lambda m, a, o, k=k: store.__setitem__(k, o)))
    try:
        for i in range(0, len(Xs), bs):
            x = Xs[i:i+bs].to("cuda", non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
            x = F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True).clamp_(0, 1)
            if v == "hflip": x = x.flip(3)
            t = model.forward_features(((x - MEAN) / STD).to(torch.bfloat16))
            res = {k: pool(store[k]) for k in LIDX} if layers else {"final": pool(t)}
            for k, a in res.items(): outs.setdefault(k, []).append(a)
            if i == 0: print("  first batch ok", {k: a.shape[1] for k, a in res.items()}, flush=True)
    finally:
        for h in hooks: h.remove()
    return {k: np.concatenate(a) for k, a in outs.items()}
log_p = f"{OUT}/log.json"; log = json.load(open(log_p)) if os.path.exists(log_p) else {}
for job, size, bs in JOBS:
    tags = list(LAYERS) if job == "layers" else [job]
    for split, views in VIEWS.items():
        for v in views:
            fs = {t: f"{OUT}/vit7b_{t}_{split}_{v}.npy" for t in tags}
            if all(os.path.exists(f) for f in fs.values()): print("skip (done)", job, split, v); continue
            t0 = time.time(); R = extract(X[split], v, size, bs, job == "layers")
            for t, f in fs.items():
                A = R[t if job == "layers" else "final"]; assert np.isfinite(A).all(), (t, split, v)
                np.save(f + ".tmp.npy", A); os.replace(f + ".tmp.npy", f); log[f"vit7b_{t}_{split}_{v}"] = [list(A.shape), round(time.time() - t0)]
            json.dump(log, open(log_p, "w"), indent=1)
            print(job, split, v, f"{(time.time()-t0)/60:.1f} min", flush=True)
            torch.cuda.empty_cache()
print("done - nothing submitted")
""")

code("""
# ===== 6. zip the features on Drive for sending back
import shutil; print(shutil.make_archive(f"{DRIVE_DIR}/frozen_7b_ms_feats", "zip", f"{DRIVE_DIR}/frozen_7b_ms"))
""")

nb = {"cells": C, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("t4/colab/task4_frozen_dinov3_7b_ms_colab.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote t4/colab/task4_frozen_dinov3_7b_ms_colab.ipynb")
