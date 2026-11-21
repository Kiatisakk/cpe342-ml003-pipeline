# Builds t4/colab/task4_fullfit_colab.ipynb (Colab, A100): full-train refit of a DINOv3 ConvNeXt for Task 4.
# Same model / augmentation / schedule as t4/kaggle/task4-v4-fullfit (Kaggle T4 version); Colab extras: Drive mount,
# Colab Secrets, local-SSD data copy, bf16, TF32, checkpoint + auto-resume on Drive, outputs written straight to Drive.
import json
C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: C.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# Task 4: full-train refit, ConvNeXt-L DINOv3 (Colab A100), v3 or v4 augmentation

Trains **one model on all train images** (no folds) and predicts the test set. Nothing is submitted.

**Before you start**
1. Runtime → Change runtime type → **A100 GPU** (High-RAM on).
2. Colab sidebar 🔑 **Secrets**, turn on "Notebook access" for:
   - `HF_TOKEN`: Hugging Face token of an account that accepted the DINOv3 licence (required).
   - `KAGGLE_USERNAME`, `KAGGLE_KEY`: only if the data comes from Kaggle (option A below).
3. Data, pick one:
   - **A** Kaggle dataset `kiatisakkk/task4-image-classification` (the owner must share it with your Kaggle account).
   - **B** `task4_data.zip` uploaded to Google Drive at `MyDrive/cpe342_t4/task4_data.zip` (used automatically if present).
4. Run all (Runtime → Run all). If Colab disconnects, reconnect and **Run all again**: training resumes from the last
   checkpoint on Drive.

**Output** (in Drive `MyDrive/cpe342_t4/fullfit/preds/`): `v4f_<backbone>_s<seed>_test_full_{plain,hflip,up}.npy`,
`..._log.json`, `test_ids.csv`. Send the whole `preds` folder (or zip it) back.

Rule notes: train labels only; the test set is only predicted (TTA flips / zoom are allowed by the TA).
""")

code("""
# ===== 1. settings (edit here only)
BACKBONE    = "cnxt_l_d3c"     # cnxt_l_d3c = ConvNeXt-L (OOF .9917) | cnxt_b_d3 | eva02_b | eva02_l | vit_l_d3 (DINOv3 ViT-L)
RECIPE      = "v3"             # "v3" = v3 augmentation (strong geometry + colour, channel shuffle, invert; class-weighted CE, ls .1)
                               # "v4" = v4 augmentation (mild geometry, sqrt sampler, ls .05; includes the JPEG-like aug)
SEEDS       = [42]             # one full-train model per seed (A100 80 GB: ~30-45 min each for Large)
FULL_EPOCHS = 14               # v3 used 14 epochs; v4 runs used 16
DATA_SOURCE = "auto"           # "auto" = Drive zip if present, else Kaggle | "drive" (option B) | "kaggle" (option A)
DRIVE_DIR   = "/content/drive/MyDrive/cpe342_t4"
CKPT_EVERY  = 2                # save a resume checkpoint to Drive every N epochs
""")

code("""
# ===== 2. GPU + packages
import subprocess, torch
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> A100"
GPU_NAME = torch.cuda.get_device_name(0); GPU_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
if "A100" not in GPU_NAME and "H100" not in GPU_NAME: print(f"WARNING: {GPU_NAME} is not an A100; training will be slower")
import importlib.metadata as im
try: tv = im.version("timm")
except im.PackageNotFoundError: tv = None
if tv != "1.0.30": subprocess.run(["pip", "install", "-q", "timm==1.0.30"], check=True)   # same timm as the Kaggle runs
import timm; print("torch", torch.__version__, "| timm", timm.__version__, "| GPU", GPU_NAME, f"{GPU_GB:.0f} GB")
""")

code("""
# ===== 3. Drive + secrets (tokens are read from Colab Secrets, never typed into the notebook)
import os
from google.colab import drive, userdata
drive.mount("/content/drive")
OUT_DIR, CKPT_DIR = f"{DRIVE_DIR}/fullfit/preds", f"{DRIVE_DIR}/fullfit/ckpt"
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
class SetupError(Exception): pass
def secret(name, required=True):
    try: return userdata.get(name)
    except Exception as e:
        if required: raise SetupError(f"add Colab secret {name} (sidebar key icon) and turn on Notebook access [{type(e).__name__}]") from None
        return None
os.environ["HF_TOKEN"] = secret("HF_TOKEN"); print("HF_TOKEN loaded")
ZIP = f"{DRIVE_DIR}/task4_data.zip"
if DATA_SOURCE == "auto": DATA_SOURCE = "drive" if os.path.exists(ZIP) else "kaggle"
if DATA_SOURCE == "kaggle":
    u, k = secret("KAGGLE_USERNAME", False), secret("KAGGLE_KEY", False)
    if not (u and k): raise SetupError(f"no data: upload task4_data.zip to {ZIP} (easiest), or add Colab secrets KAGGLE_USERNAME + KAGGLE_KEY")
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = u, k; print("Kaggle credentials loaded")
print("data source:", DATA_SOURCE)
""")

code("""
# ===== 4. data -> local SSD (/content/data), never trained from Drive directly
import os, glob, zipfile, subprocess, time
DATA = "/content/data"; os.makedirs(DATA, exist_ok=True); t0 = time.time()
if not glob.glob(f"{DATA}/**/train.csv", recursive=True):
    if DATA_SOURCE == "kaggle":
        subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", "kiatisakkk/task4-image-classification", "-p", DATA, "--unzip"], check=True)
    else:
        zipfile.ZipFile(ZIP).extractall(DATA)                  # copy to local SSD first, then train from there
hits = [os.path.dirname(p) for p in glob.glob(f"{DATA}/**/train.csv", recursive=True)]
ROOT = [h for h in hits if os.path.isdir(f"{h}/train") and os.path.isdir(f"{h}/test")][0]
print("data at", ROOT, f"({time.time()-t0:.0f}s)")
""")

code("""
# ===== 5. config, data arrays, model, augmentation (same as the Kaggle v4 recipe)
import os, time, gc, random, json, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
import timm.optim, timm.utils, timm.data              # submodules are not always imported by "import timm"
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")
PRESETS = {
    "cnxt_l_d3c": dict(names=["convnext_large.dinov3_lvd1689m"], size=(224, 416), batch=32, lr=1.5e-4, llrd=0.8, dpr=0.2),
    "cnxt_b_d3":  dict(names=["convnext_base.dinov3_lvd1689m"],  size=(224, 416), batch=32, lr=1.5e-4, llrd=0.8, dpr=0.2),
    "eva02_b":    dict(names=["eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"], size=(224, 420), batch=32, lr=7e-5, llrd=0.75, dpr=0.15),
    "eva02_l":    dict(names=["eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"], size=(224, 420), batch=32, lr=4e-5, llrd=0.8, dpr=0.2),
    "vit_l_d3":   dict(names=["vit_large_patch16_dinov3.lvd1689m"], size=(224, 416), batch=32, lr=5e-5, llrd=0.75, dpr=0.2),
}
IS_VIT = not BACKBONE.startswith("cnxt"); STRIDE = 14 if BACKBONE.startswith("eva") else 16 if IS_VIT else 32
PR = PRESETS[BACKBONE]; IMG_H, IMG_W = PR["size"]; BATCH, LR, LLRD = PR["batch"], PR["lr"], PR["llrd"]
WD, LPFT_EPOCHS, LABEL_SMOOTH, EMA_DECAY, HFLIP_P = 0.05, 1, 0.05, 0.9995, 0.5
GRAD_CKPT = GPU_GB < 60 or BACKBONE == "eva02_l"           # 40 GB A100 or EVA02-L: checkpointing on; else off (faster)
TTA_VIEWS = {"plain": (False, 1.0), "hflip": (True, 1.0), "up": (False, 1.125)}
DEVICE = "cuda"; AMP = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

train = pd.read_csv(f"{ROOT}/train.csv"); test = pd.read_csv(f"{ROOT}/test.csv")
assert list(train.columns[:2]) == ["id", "class"], "expects the dataset's train.csv (id,class)"
CLASSES = sorted(train["class"].unique()); cid = {c: i for i, c in enumerate(CLASSES)}; K = len(CLASSES)
assert CLASSES == ["7-11k", "fiveN", "freefried", "pube", "roblock"], CLASSES
y = train["class"].map(cid).values.astype(np.int64)
def load_all(folder, ids, name):
    p = f"/content/cache_{name}.npy"
    if os.path.exists(p): return np.load(p)
    t0 = time.time(); read = lambda i: np.asarray(Image.open(f"{folder}/{i}.jpg").convert("RGB"))
    with ThreadPoolExecutor(32) as ex: arr = np.stack(list(ex.map(read, ids, chunksize=64)))
    np.save(p, arr); print(f"{name}: {arr.shape} in {time.time()-t0:.0f}s", flush=True); return arr
TR = torch.from_numpy(load_all(f"{ROOT}/train", train["id"], "train")).pin_memory()
TE = torch.from_numpy(load_all(f"{ROOT}/test", test["id"], "test")).pin_memory()
pd.DataFrame({"id_img": test["id"]}).to_csv(f"{OUT_DIR}/test_ids.csv", index=False)   # row order of every test file

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def create_model():
    kw = dict(img_size=(IMG_H, IMG_W), dynamic_img_size=True) if IS_VIT else {}       # ViT: interpolate pos-embed to our size
    m = timm.create_model(PR["names"][0], pretrained=True, num_classes=K, drop_path_rate=PR["dpr"], **kw)
    if GRAD_CKPT: m.set_grad_checkpointing(True)
    return m
_m = create_model(); cfg = timm.data.resolve_model_data_config(_m)
MEAN = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1); STD = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)
print("model", PR["names"][0], f"{sum(p.numel() for p in _m.parameters())/1e6:.0f}M params | grad ckpt", GRAD_CKPT, "| amp", AMP); del _m

def rup(v): return int(round(v / STRIDE)) * STRIDE
def to_input(xb, scale=1.0):
    x = xb.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    size = (IMG_H, IMG_W) if scale == 1.0 else (rup(IMG_H * scale), rup(IMG_W * scale))
    return F.interpolate(x, size=size, mode="bicubic", align_corners=False).clamp_(0, 1)
def gpu_augment(x):                                        # identical to the Kaggle v4 recipe
    B, _, H, W = x.shape; dev = x.device
    ang = torch.where(torch.rand(B, device=dev) < 0.5, torch.rand(B, device=dev) * 16 - 8, torch.zeros(B, device=dev)) * torch.pi / 180
    s = torch.empty(B, device=dev).uniform_(0.92, 1.08); c, sn = torch.cos(ang) / s, torch.sin(ang) / s
    fl = torch.where(torch.rand(B, device=dev) < HFLIP_P, -1.0, 1.0)
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = c * fl; th[:, 0, 1] = -sn * H / W; th[:, 1, 0] = sn * W / H * fl; th[:, 1, 1] = c
    th[:, :, 2] = torch.empty(B, 2, device=dev).uniform_(-0.04, 0.04)
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="reflection", align_corners=False)
    x = x * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.75, 1.25)
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.75, 1.25) + m
    g = x.mean(1, keepdim=True); x = (x - g) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.6, 1.4) + g
    gr = torch.rand(B, device=dev) < 0.05; x[gr] = x[gr].mean(1, keepdim=True).expand(-1, 3, -1, -1)
    lo = torch.rand(B, device=dev) < 0.25
    if lo.any():
        f = float(np.random.uniform(0.4, 0.75))
        x[lo] = F.interpolate(F.interpolate(x[lo], scale_factor=f, mode="bilinear", align_corners=False), size=(H, W), mode="bilinear", align_corners=False)
    nz = (torch.rand(B, 1, 1, 1, device=dev) < 0.3).float() * torch.empty(B, 1, 1, 1, device=dev).uniform_(0, 0.05)
    x = (x + torch.randn_like(x) * nz).clamp_(0, 1)
    for i in torch.nonzero(torch.rand(B, device=dev) < 0.25).flatten().tolist():
        eh, ew = int(H * np.random.uniform(0.1, 0.3)), int(W * np.random.uniform(0.1, 0.3))
        y0, x0 = np.random.randint(0, H - eh), np.random.randint(0, W - ew); x[i, :, y0:y0+eh, x0:x0+ew] = torch.rand(3, 1, 1, device=dev)
    return x
def gpu_augment_v3(x):                                     # the v3 (sub_v3_5f) augmentation, written before any test EDA
    B, _, H, W = x.shape; dev = x.device
    ang = torch.where(torch.rand(B, device=dev) < 0.7, (torch.rand(B, device=dev) * 60 - 30), torch.zeros(B, device=dev)) * torch.pi / 180
    s = torch.empty(B, device=dev).uniform_(0.85, 1.2); c, sn = torch.cos(ang) / s, torch.sin(ang) / s
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = c; th[:, 0, 1] = -sn * H / W; th[:, 1, 0] = sn * W / H; th[:, 1, 1] = c
    th[:, :, 2] = torch.empty(B, 2, device=dev).uniform_(-0.08, 0.08)
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False)
    x = x * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.7, 1.3) + m
    g = x.mean(1, keepdim=True); x = (x - g) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.5, 1.6) + g
    pm = torch.rand(B, device=dev) < 0.25
    if pm.any():
        idx = torch.stack([torch.randperm(3, device=dev) for _ in range(int(pm.sum()))])
        x[pm] = torch.gather(x[pm], 1, idx[:, :, None, None].expand(-1, -1, H, W))
    inv = torch.rand(B, device=dev) < 0.1; x[inv] = 1 - x[inv]
    gr = torch.rand(B, device=dev) < 0.1; x[gr] = x[gr].mean(1, keepdim=True).expand(-1, 3, -1, -1)
    nz = (torch.rand(B, 1, 1, 1, device=dev) < 0.5).float() * torch.empty(B, 1, 1, 1, device=dev).uniform_(0, 0.08)
    return (x + torch.randn_like(x) * nz).clamp_(0, 1)
AUG = gpu_augment_v3 if RECIPE == "v3" else gpu_augment
norm = lambda x: (x - MEAN) / STD
def sample_weights(idx):                                   # v4: sqrt-inverse class frequency sampler; v3: plain shuffle + class-weighted loss
    if RECIPE == "v3": return np.full(len(idx), 1 / len(idx))
    cnt = np.bincount(y[idx], minlength=K).astype(float); w = (1 / np.sqrt(cnt))[y[idx]]; return w / w.sum()
CNT = np.bincount(y, minlength=K)
CRIT_W = torch.tensor(len(y) / (K * CNT), dtype=torch.float32, device=DEVICE) if RECIPE == "v3" else None
LS = 0.1 if RECIPE == "v3" else LABEL_SMOOTH
@torch.no_grad()
def predict(model, X, flip=False, scale=1.0, bs=256):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        x = to_input(X[i:i+bs], scale)
        if flip: x = x.flip(3)
        with torch.autocast("cuda", dtype=AMP): out.append(torch.softmax(model(norm(x)).float(), 1).cpu())
    return torch.cat(out).numpy()
""")

code("""
# ===== 6. train on ALL train rows, one model per seed; checkpoint + resume on Drive
def ckpt_path(tag): return f"{CKPT_DIR}/{tag}.pt"
def train_full(seed, tag):
    seed_all(seed); idx = np.arange(len(y))
    model = create_model().to(DEVICE).to(memory_format=torch.channels_last)
    head = set(id(p) for p in model.get_classifier().parameters())
    opt = timm.optim.create_optimizer_v2(model, opt="adamw", lr=LR / LLRD, weight_decay=WD, layer_decay=LLRD)
    for g in opt.param_groups:
        g["lr"] = g["lr"] * g.get("lr_scale", 1.0)
        if any(id(p_) in head for p_ in g["params"]): g["lr"] = LR * 10
    spe = len(idx) // BATCH; steps = FULL_EPOCHS * spe; warm = max(1, int(0.05 * steps)); lp = LPFT_EPOCHS * spe
    def lr_lambda(s):
        if s < lp: return min(1.0, (s + 1) / warm)
        if s < lp + spe: return 0.05 + 0.95 * (s - lp + 1) / spe
        t = (s - lp - spe) / max(1, steps - lp - spe); return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP == torch.float16))
    crit = nn.CrossEntropyLoss(weight=CRIT_W, label_smoothing=LS)
    ema = timm.utils.ModelEmaV3(model, decay=EMA_DECAY); yt = torch.from_numpy(y); sw = sample_weights(idx)
    start, hist = 0, []
    if os.path.exists(ckpt_path(tag)):
        ck = torch.load(ckpt_path(tag), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.module.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start, hist = ck["epoch"], ck["hist"]
        np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"]); print(f"resumed {tag} at epoch {start}", flush=True)
    for ep in range(start, FULL_EPOCHS):
        model.train(); t0 = time.time(); tot = 0
        perm = idx[np.random.choice(len(idx), len(idx), replace=True, p=sw)]
        frozen = ep < LPFT_EPOCHS
        for p_ in model.parameters():
            if id(p_) not in head: p_.requires_grad_(not frozen)
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            b = torch.from_numpy(perm[i:i+BATCH])
            x = norm(AUG(to_input(TR[b]))).contiguous(memory_format=torch.channels_last); yb = yt[b].to(DEVICE)
            with torch.autocast("cuda", dtype=AMP): loss = crit(model(x), yb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_([p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model); tot += loss.item()
        hist.append((ep + 1, round(tot / max(1, len(perm) // BATCH), 4), round(time.time() - t0)))
        print(f"{tag} ep {ep+1}/{FULL_EPOCHS} loss {hist[-1][1]} ({hist[-1][2]}s, peak GPU {torch.cuda.max_memory_allocated()/2**30:.0f} GB)", flush=True)
        if (ep + 1) % CKPT_EVERY == 0 and ep + 1 < FULL_EPOCHS:
            tmp = ckpt_path(tag) + ".tmp"
            torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": ep + 1, "hist": hist, "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}, tmp)
            os.replace(tmp, ckpt_path(tag))
    pt = {v: predict(ema.module, TE, *TTA_VIEWS[v]) for v in TTA_VIEWS}
    del model, opt, ema; gc.collect(); torch.cuda.empty_cache()
    return pt, hist

for seed in SEEDS:
    tag = f"v4f_{BACKBONE}_{RECIPE}aug_s{seed}"
    if os.path.exists(f"{OUT_DIR}/{tag}_log.json"): print(tag, "already done - skipped"); continue
    t0 = time.time(); pt, hist = train_full(seed, tag)
    for v in TTA_VIEWS: np.save(f"{OUT_DIR}/{tag}_test_full_{v}.npy", pt[v])
    P = np.mean(list(pt.values()), 0)
    json.dump({"model": PR["names"][0], "config": PR, "recipe": RECIPE, "seed": seed, "epochs": FULL_EPOCHS, "hist": hist, "gpu": GPU_NAME,
               "minutes": round((time.time() - t0) / 60, 1), "test_rows": len(TE),
               "test_share": dict(zip(CLASSES, (np.bincount(P.argmax(1), minlength=K) / len(P)).round(3).tolist()))},
              open(f"{OUT_DIR}/{tag}_log.json", "w"), indent=1)
    if os.path.exists(ckpt_path(tag)): os.remove(ckpt_path(tag))          # free Drive space once the seed is done
    print(f"== {tag} done in {(time.time()-t0)/60:.0f} min", flush=True)
print("all done - files in", OUT_DIR, "(nothing submitted)")
""")

code("""
# ===== 7. (optional) zip the outputs on Drive for sending back
import shutil; print(shutil.make_archive(f"{DRIVE_DIR}/fullfit_preds", "zip", OUT_DIR))
""")

nb = {"cells": C, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
import copy
VARIANTS = {   # file -> (BACKBONE, RECIPE, FULL_EPOCHS, SEEDS, note); only the settings cell differs
    "task4_fullfit_v3aug_colab.ipynb":   ("cnxt_l_d3c", "v3", 14, "[42]", "ConvNeXt-L, v3 augmentation; A100 80 GB ~45 min, 40 GB ~1 h, L4 ~3-3.5 h"),
    "task4_fullfit_eva02l_colab.ipynb":  ("eva02_l", "v4", 14, "[42]", "EVA02-L (BAAI); A100 ~2-2.5 h, L4 too slow"),
    "task4_fullfit_vitl_d3_colab.ipynb": ("vit_l_d3", "v4", 14, "[42]", "DINOv3 ViT-L; A100 ~1.5-2 h"),
}
for fn, (bb, rc, ep, sd, note) in VARIANTS.items():
    v = copy.deepcopy(nb); src = "".join(v["cells"][1]["source"])
    src = src.replace('BACKBONE    = "cnxt_l_d3c"', f'BACKBONE    = "{bb}"').replace('RECIPE      = "v3"', f'RECIPE      = "{rc}"')
    src = src.replace("FULL_EPOCHS = 14", f"FULL_EPOCHS = {ep}").replace("SEEDS       = [42]", f"SEEDS       = {sd}")
    v["cells"][1]["source"] = ("# " + note + chr(10) + src).splitlines(True)
    json.dump(v, open(f"t4/colab/{fn}", "w", encoding="utf-8"), indent=1, ensure_ascii=False); print("wrote", fn)
