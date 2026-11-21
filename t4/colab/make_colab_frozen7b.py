# Builds t4/colab/task4_frozen_dinov3_7b_colab.ipynb: frozen DINOv3 ViT-7B features for Task 4 (Colab A100).
# Same output format as the Kaggle frozen-feature kernels ([CLS, mean patch], float16, feats/{tag}_{split}_{view}.npy +
# train_names.npy / test_names.npy), so the local head script works unchanged:
#   python work/t4_head_lowmem.py t4/preds/frozen_7b/feats vit_7b_patch16_dinov3 plain,hflip _ph
import json
C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: C.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# Task 4: frozen DINOv3 ViT-7B features (Colab A100)

Extracts features only (no training, nothing submitted). Train views plain + hflip, test views plain + hflip.

**Before you start**
1. Runtime → **A100 GPU** + **High-RAM** (the 7B weights need ~30 GB of CPU RAM while loading).
2. 🔑 Secrets: `HF_TOKEN` of an account that accepted the DINOv3 licence, Notebook access on.
3. `task4_data.zip` in Drive at `MyDrive/cpe342_t4/` (same file as the full-train notebooks).
4. Run all. Time: ~1-1.5 h on an A100. Each view is saved to Drive as soon as it is done, so after a disconnect
   **Run all** again skips the finished views.

**Output**: `MyDrive/cpe342_t4/frozen_7b/feats/` (~2 GB) and `MyDrive/cpe342_t4/frozen_7b_feats.zip`. Send the zip back.
""")

code("""
# ===== 1. settings
MODEL    = "vit_7b_patch16_dinov3.lvd1689m"
IMG_H, IMG_W = 224, 416                     # same size as the other DINOv3 feature runs (patch 16)
VIEWS    = {"test": ["plain", "hflip"], "train": ["plain", "hflip"]}
BATCH    = 32
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
OUT = f"{DRIVE_DIR}/frozen_7b/feats"; os.makedirs(OUT, exist_ok=True)
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
def load_all(folder, names):
    read = lambda i: np.asarray(Image.open(f"{folder}/{i}.jpg").convert("RGB").resize((230, 120)))
    with ThreadPoolExecutor(32) as ex: return torch.from_numpy(np.stack(list(ex.map(read, names, chunksize=64))))
t0 = time.time(); X = {"train": load_all(f"{ROOT}/train", tr_names), "test": load_all(f"{ROOT}/test", te_names)}
print({k: tuple(v.shape) for k, v in X.items()}, f"{time.time()-t0:.0f}s")
""")

code("""
# ===== 4. model (bf16 on the GPU)
import torch.nn.functional as F, gc
t0 = time.time()
model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=(IMG_H, IMG_W), dynamic_img_size=True)
cfg = timm.data.resolve_model_data_config(model)
model = model.to(torch.bfloat16).to("cuda").eval(); gc.collect()
MEAN = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1); STD = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
TAG = MODEL.split(".")[0]
print(TAG, f"{sum(p.numel() for p in model.parameters())/1e9:.1f}B params | prefix tokens {model.num_prefix_tokens} | "
      f"GPU {torch.cuda.memory_allocated()/2**30:.0f} GB | {time.time()-t0:.0f}s")
""")

code("""
# ===== 5. extract [CLS, mean patch] per view; each view saved to Drive (re-running skips finished views)
import json
@torch.no_grad()
def extract(Xs, v):
    out = []
    for i in range(0, len(Xs), BATCH):
        x = Xs[i:i+BATCH].to("cuda", non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        x = F.interpolate(x, size=(IMG_H, IMG_W), mode="bicubic", align_corners=False).clamp_(0, 1)
        if v == "hflip": x = x.flip(3)
        t = model.forward_features(((x - MEAN) / STD).to(torch.bfloat16)).float()
        out.append(torch.cat([t[:, 0], t[:, model.num_prefix_tokens:].mean(1)], 1).cpu().numpy().astype(np.float16))
        if i == 0: print(f"  first batch ok, feature dim {out[0].shape[1]}", flush=True)
    return np.concatenate(out)
log_p = f"{OUT}/log.json"; log = json.load(open(log_p)) if os.path.exists(log_p) else {}
for split, views in VIEWS.items():
    for v in views:
        f = f"{OUT}/{TAG}_{split}_{v}.npy"
        if os.path.exists(f): print("skip (done)", f); continue
        t0 = time.time(); A = extract(X[split], v); np.save(f + ".tmp.npy", A); os.replace(f + ".tmp.npy", f)
        log[f"{TAG}_{split}_{v}"] = [list(A.shape), round(time.time() - t0)]; json.dump(log, open(log_p, "w"), indent=1)
        print(TAG, split, v, A.shape, f"{(time.time()-t0)/60:.1f} min", flush=True)
print("done - nothing submitted")
""")

code("""
# ===== 6. zip the features on Drive for sending back
import shutil; print(shutil.make_archive(f"{DRIVE_DIR}/frozen_7b_feats", "zip", f"{DRIVE_DIR}/frozen_7b"))
""")

nb = {"cells": C, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("t4/colab/task4_frozen_dinov3_7b_colab.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote t4/colab/task4_frozen_dinov3_7b_colab.ipynb")
