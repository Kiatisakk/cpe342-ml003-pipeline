# task4-v4-fullfit-l3: ConvNeXt-L DINOv3 full-train with the v3 augmentation recipe (T4_RECIPE=v3), 14 epochs, seed 42.
# Kaggle T4: ~5.5 h. Needs GPU T4 + Internet + Secret HF_TOKEN (start from the UI). Outputs v4f_cnxt_l_d3c_v3aug_s42_test_full_*.npy
# Task 4 - FULL-TRAIN refit of a DINOv3 ConvNeXt (no folds). Same model code as task4-v4-cnxtl-d3.
# Why: every fine-tuned model in the blend is a 3-fold model that saw only 2/3 of train. The test is new scenes, so more
# training scenes should help. Each run trains on ALL train rows for a FIXED number of epochs (taken from the 3-fold runs'
# best epochs: Base 17/12, Colab Large 17/11/15 -> 16), no early stopping, final EMA model. Train labels only; the test
# is only predicted (TTA plain/hflip/up), never used for any choice.
# T4_LOWQ_AUG (default "0" = OFF since 16 Nov: the JPEG-like degradation aug was added because test looked lower quality,
# i.e. chosen from test EDA, which the TA ruled out; output tag gets suffix _nolq). Defaults now: cnxt_b_d3, 14 epochs, seed 42.
# Settings: T4_BACKBONE (cnxt_l_d3c = Colab Large recipe, batch 16 x accum 2 = 32 to fit a T4 | cnxt_b_d3),
# T4_SEEDS (default "42"; several seeds run one after another, time guard 10.6 h), T4_FULL_EPOCHS (default 16).
# Expected time on a Kaggle T4 for Large: ~7.5 h (the 3-fold Kaggle L run took ~17 min/epoch on 2/3 of train).
# Outputs /kaggle/working/preds/v4f_<backbone>_s<seed>_test_full_{plain,hflip,up}.npy + _log.json. Nothing is submitted.
# Run from the Kaggle UI: GPU T4, Internet ON, Add-ons -> Secrets -> HF_TOKEN attached.
# Task 4 — ConvNeXt-L DINOv3 fine-tune (preset cnxt_l_d3). Same code as task4-v4-cnxts-in22k; only the preset and guard differ.
# Why: on the shifted test, every bigger DINOv3 ConvNeXt helped the LB (T -> +S v28 +0.0012 -> +B v30 +0.0026).
# ~5 h per fold on a T4 -> about 2 folds per 12 h session; per-fold files are saved after each fold, so later sessions run
# the rest with T4_FOLDS (edit the default below) and the blend script merges dirs passed with the same tag.
# Needs HF_TOKEN (DINOv3 is gated) -> start from the Kaggle UI with the secret attached, GPU T4, Internet on. NO SUBMIT.
# Task 4 v4 — Kaggle GPU training script (based on t4/code/task4_5fold.py = v3_5f, OOF macro-F1 0.9889)
# NOT run in the repo sandbox (no GPU / no images). Paste into a Kaggle notebook (GPU T4 x1 or P100, Internet ON).
# NO SUBMIT - writes /kaggle/working/preds only.
#
# Changes vs v3_5f (see t4/research/REPORT.md):
#   1. BACKBONE is a preset key: run the notebook once per backbone, then blend runs with t4_blend_runs.py.
#        "cnxt_t_d3"  convnext_tiny.dinov3   (v3 model, at higher res, as the control)
#        "cnxt_s_d3"  convnext_small.dinov3  (bigger DINOv3 ConvNeXt)
#        "eva02_s"    eva02_small_patch14_336.mim_in22k_ft_in1k (ViT family -> diversity for the 2-model ensemble)
#   2. Higher input resolution, native 230x120 aspect ratio kept (v3: 192x352, rotation +-30 deg with black borders
#      and scale 0.85-1.2 = aggressive crop). v4: milder geometry (rot +-8, scale 0.92-1.08, shift 4%) + optional hflip.
#   3. Class-balanced sampling (sqrt-inverse-frequency by default, train labels only) instead of weighted CE.
#      Optional "prior" mode samples to a user-given class mix; leave it OFF unless the team agrees it is allowed
#      (the test mix is only known from model predictions on test; competition rule: no test usage for training).
#   4. Longer schedule (20 epochs) and NO best-epoch-on-validation selection: the final EMA model is used, so the
#      OOF score is an honest estimate (v3 picked the best of 14 epochs on the same fold it reports).
#   5. TTA: every view (plain, hflip, up-scale) is saved separately for OOF and test, so the TTA combination can be
#      chosen on OOF afterwards (t4_blend_runs.py) instead of guessing.
#   6. Time guard for the 12 h Kaggle session; RUN_FOLDS can be split over several sessions (files are per fold).
# Output format is the same as v3 (+ per-view files):
#   preds/{TAG}_oof.npy (N_train x 5, mean of TTA_DEFAULT views), preds/{TAG}_oof_{view}.npy,
#   preds/{TAG}_test_fold{k}.npy, preds/{TAG}_test_fold{k}_{view}.npy, preds/task4_{TAG}_test_proba.npy,
#   preds/task4_{TAG}.csv (id,id_img,pred_num,pred_str), class order = datacard/alphabetical (0 7-11k .. 4 roblock).
import os, glob, time, gc, random, warnings, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
import timm.optim, timm.utils
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
warnings.filterwarnings("ignore")

# ============================================================== config
BACKBONE   = os.environ.get("T4_BACKBONE", "cnxt_l_d3c")       # cnxt_t_d3 | cnxt_s_d3 | eva02_s
RUN_FOLDS  = [int(f) for f in os.environ.get("T4_FOLDS", "0,1,2").split(",")]
PRESETS = {   # names tried in order; (H, W) must suit the stride (32 for ConvNeXt, 14 for EVA02)
    "cnxt_t_d3": dict(names=["convnext_tiny.dinov3_lvd1689m", "convnext_tiny.fb_in22k_ft_in1k"],
                      size=(256, 480), batch=48, lr=2e-4, epochs=20, llrd=0.8, dpr=0.1, need_hf=True),
    "cnxt_s_d3": dict(names=["convnext_small.dinov3_lvd1689m"],
                      size=(224, 416), batch=32, lr=1.5e-4, epochs=18, llrd=0.8, dpr=0.2, need_hf=True),
    "cnxt_b_d3": dict(names=["convnext_base.dinov3_lvd1689m"],
                      size=(224, 416), batch=32, lr=1.5e-4, epochs=18, llrd=0.8, dpr=0.2, need_hf=True),
    "cnxt_l_d3c": dict(names=["convnext_large.dinov3_lvd1689m"],                     # Colab recipe (OOF .9917): effective batch 32
                      size=(224, 416), batch=16, accum=2, lr=1.5e-4, epochs=18, llrd=0.8, dpr=0.2, need_hf=True, grad_ckpt=True),
    "cnxt_l_d3": dict(names=["convnext_large.dinov3_lvd1689m"],
                      size=(224, 416), batch=16, lr=1e-4, epochs=16, llrd=0.8, dpr=0.3, need_hf=True, grad_ckpt=True),
    "cnxt_s_in22k": dict(names=["convnext_small.fb_in22k_ft_in1k"],
                      size=(224, 416), batch=32, lr=1.5e-4, epochs=18, llrd=0.8, dpr=0.2, need_hf=False),
    "eva02_s":   dict(names=["eva02_small_patch14_336.mim_in22k_ft_in1k"],
                      size=(224, 420), batch=32, lr=1e-4, epochs=18, llrd=0.75, dpr=0.1, need_hf=False),
}
PR = PRESETS[BACKBONE]
RUN_TAG    = f"v4_{BACKBONE}"
N_FOLDS, SEED = 3, 42                                          # 3 folds (OOF covers all rows -> still blendable with v3 on OOF)
IMG_H, IMG_W = PR["size"]; BATCH, LR, EPOCHS, LLRD = PR["batch"], PR["lr"], PR["epochs"], PR["llrd"]
WD, LPFT_EPOCHS, LABEL_SMOOTH, EMA_DECAY = 0.05, 1, 0.05, 0.9995
HFLIP_P    = 0.5                                               # set 0.0 if OOF of the hflip view is clearly worse
SAMPLER    = "sqrt"                                            # none | sqrt | balanced | prior
PRIOR_MIX  = None                                              # only for SAMPLER="prior": list of 5 floats (datacard order)
TTA_VIEWS  = {"plain": (False, 1.0), "hflip": (True, 1.0), "up": (False, 1.125)}
TTA_DEFAULT = ["plain", "hflip"]                               # views averaged into {TAG}_oof.npy / test_proba
ES_FRAC, ES_PATIENCE, ES_MIN_EP = 0.10, 3, 6                # early stop on an INNER holdout of the train fold (never the val fold)
TIME_LIMIT_H = 10.6                                            # stop starting new folds after this (Kaggle limit 12 h)
OUT, CACHE = "/kaggle/working", "/tmp/t4cache"
T_START = time.time()

# ============================================================== secrets (never hardcode tokens)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN"); print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as e:
    if PR["need_hf"]: raise SystemExit(f"STOP: attach Kaggle secret HF_TOKEN (Add-ons -> Secrets) first ({type(e).__name__})")
    print("no HF_TOKEN (not needed for", BACKBONE, ")")

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED); DEVICE = "cuda"
assert torch.cuda.is_available(), "turn on GPU"
torch.backends.cudnn.benchmark = True
print(torch.cuda.get_device_name(0), "| timm", timm.__version__, "| backbone", BACKBONE, PR, flush=True)

# ============================================================== data (same discovery as v3)
root = None
for _ in range(30):
    hits = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train.csv", recursive=True)]
    hits = [h for h in hits if os.path.isdir(f"{h}/train") and os.path.isdir(f"{h}/test") and os.path.isfile(f"{h}/test.csv")]
    if hits: root = sorted(hits)[0]; break
    time.sleep(10)
assert root, "task4 image data not found"
SAMPLE = sorted(glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True))
train = pd.read_csv(f"{root}/train.csv"); test = pd.read_csv(f"{root}/test.csv")
CLASSES = sorted(train["class"].unique()); cid = {c: i for i, c in enumerate(CLASSES)}; K = len(CLASSES)
assert CLASSES == ["7-11k", "fiveN", "freefried", "pube", "roblock"], CLASSES
y = train["class"].map(cid).values.astype(np.int64)
print(root, train.shape, test.shape, CLASSES, flush=True)

def load_all(folder, ids, name):
    os.makedirs(CACHE, exist_ok=True); p = f"{CACHE}/{name}.npy"
    if os.path.exists(p): return np.load(p)
    t0 = time.time(); read = lambda i: np.asarray(Image.open(f"{folder}/{i}.jpg").convert("RGB"))
    with ThreadPoolExecutor(32) as ex: arr = np.stack(list(ex.map(read, ids, chunksize=64)))
    np.save(p, arr); print(f"{name}: {arr.shape} in {time.time()-t0:.0f}s", flush=True); return arr
TR = torch.from_numpy(load_all(f"{root}/train", train["id"], "train"))     # uint8 N x 120 x 230 x 3
TE = torch.from_numpy(load_all(f"{root}/test", test["id"], "test"))

# ============================================================== model
IS_VIT = BACKBONE.startswith("eva")
STRIDE = 14 if IS_VIT else 32
def create_model():
    for name in PR["names"]:
        try:
            kw = dict(pretrained=True, num_classes=K, drop_path_rate=PR["dpr"])
            if IS_VIT: kw.update(img_size=(IMG_H, IMG_W), dynamic_img_size=True)   # pos-embed interpolated to our size
            return timm.create_model(name, **kw), name
        except Exception as e: print(f"skip {name}: {type(e).__name__}: {str(e)[:150]}")
    raise RuntimeError("cannot load model - check Internet / HF_TOKEN")
_m, MODEL_NAME = create_model()
cfg = timm.data.resolve_model_data_config(_m)
MEAN = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1); STD = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)
print("model:", MODEL_NAME, f"{sum(p.numel() for p in _m.parameters())/1e6:.1f}M params", flush=True); del _m
if PR["need_hf"]: assert "dinov3" in MODEL_NAME, "DINOv3 weights not loaded (check HF_TOKEN / Internet)"

def rup(v): return int(round(v / STRIDE)) * STRIDE
def to_input(xb_u8, scale=1.0):
    x = xb_u8.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    size = (IMG_H, IMG_W) if scale == 1.0 else (rup(IMG_H * scale), rup(IMG_W * scale))
    return F.interpolate(x, size=size, mode="bicubic", align_corners=False).clamp_(0, 1)   # whole frame, no crop
def gpu_augment(x):
    B, _, H, W = x.shape; dev = x.device
    ang = torch.where(torch.rand(B, device=dev) < 0.5, torch.rand(B, device=dev) * 16 - 8, torch.zeros(B, device=dev)) * torch.pi / 180
    s = torch.empty(B, device=dev).uniform_(0.92, 1.08); c, sn = torch.cos(ang) / s, torch.sin(ang) / s
    fl = torch.where(torch.rand(B, device=dev) < HFLIP_P, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = c * fl; theta[:, 0, 1] = -sn * H / W; theta[:, 1, 0] = sn * W / H * fl; theta[:, 1, 1] = c
    theta[:, :, 2] = torch.empty(B, 2, device=dev).uniform_(-0.04, 0.04)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), mode="bilinear", padding_mode="reflection", align_corners=False)
    x = x * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.75, 1.25)                     # brightness
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.75, 1.25) + m   # contrast
    g = x.mean(1, keepdim=True); x = (x - g) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.6, 1.4) + g             # saturation
    gr = torch.rand(B, device=dev) < 0.05; x[gr] = x[gr].mean(1, keepdim=True).expand(-1, 3, -1, -1)
    # JPEG-ish degradation: down/up-sample a quarter of the batch (test screenshots look lower quality / less confident)
    lo = torch.rand(B, device=dev) < (0.25 if globals().get("LOWQ_AUG", True) else 0.0)
    if lo.any():
        f = float(np.random.uniform(0.4, 0.75))
        x[lo] = F.interpolate(F.interpolate(x[lo], scale_factor=f, mode="bilinear", align_corners=False), size=(H, W), mode="bilinear", align_corners=False)
    nz = (torch.rand(B, 1, 1, 1, device=dev) < 0.3).float() * torch.empty(B, 1, 1, 1, device=dev).uniform_(0, 0.05)
    x = (x + torch.randn_like(x) * nz).clamp_(0, 1)
    er = torch.rand(B, device=dev) < 0.25                                                  # random erasing (1 box)
    for i in torch.nonzero(er).flatten().tolist():
        eh, ew = int(H * np.random.uniform(0.1, 0.3)), int(W * np.random.uniform(0.1, 0.3))
        y0, x0 = np.random.randint(0, H - eh), np.random.randint(0, W - ew); x[i, :, y0:y0+eh, x0:x0+ew] = torch.rand(3, 1, 1, device=dev)
    return x
def gpu_augment_v3(x):                                         # the v3 (v3_5f) augmentation, written before any test EDA
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
norm = lambda x: (x - MEAN) / STD

def sample_weights(idx):
    cnt = np.bincount(y[idx], minlength=K).astype(float)
    if SAMPLER == "none": return None
    if SAMPLER == "balanced": wc = 1 / cnt
    elif SAMPLER == "sqrt": wc = 1 / np.sqrt(cnt)
    elif SAMPLER == "prior":
        assert PRIOR_MIX is not None and len(PRIOR_MIX) == K, "set PRIOR_MIX"; wc = np.asarray(PRIOR_MIX) / cnt
    else: raise ValueError(SAMPLER)
    w = wc[y[idx]]; return w / w.sum()

@torch.no_grad()
def predict(model, X, flip=False, scale=1.0, bs=128):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        x = to_input(X[i:i+bs], scale)
        if flip: x = x.flip(3)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(norm(x).contiguous(memory_format=torch.channels_last)).float().softmax(1).cpu())
    return torch.cat(out).numpy()
def predict_views(model, X): return {v: predict(model, X, *TTA_VIEWS[v]) for v in TTA_VIEWS}

def train_fold(k, tr_idx, va_idx):
    seed_all(SEED + k)
    from sklearn.model_selection import train_test_split
    tr_idx, es_idx = train_test_split(tr_idx, test_size=ES_FRAC, stratify=y[tr_idx], random_state=SEED + k)
    best_nll, best_ep, best_state, bad = 1e9, -1, None, 0
    model, _ = create_model(); model = model.to(DEVICE).to(memory_format=torch.channels_last)
    if PR.get("grad_ckpt"): model.set_grad_checkpointing(True)          # L: trade ~30% speed for activation memory (T4 16 GB)
    print(f"fold {k}: GPU mem after model {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    head = set(id(p) for p in model.get_classifier().parameters())
    opt = timm.optim.create_optimizer_v2(model, opt="adamw", lr=LR / LLRD, weight_decay=WD, layer_decay=LLRD)
    for g in opt.param_groups:
        g["lr"] = g["lr"] * g.get("lr_scale", 1.0)
        if any(id(p_) in head for p_ in g["params"]): g["lr"] = LR * 10
    spe = len(tr_idx) // BATCH; steps = EPOCHS * spe; warm = max(1, int(0.05 * steps)); lp = LPFT_EPOCHS * spe
    def lr_lambda(s):
        if s < lp: return min(1.0, (s + 1) / warm)
        if s < lp + spe: return 0.05 + 0.95 * (s - lp + 1) / spe
        t = (s - lp - spe) / max(1, steps - lp - spe); return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda); scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)              # class balance handled by the sampler
    ema = timm.utils.ModelEmaV3(model, decay=EMA_DECAY); yt = torch.from_numpy(y); sw = sample_weights(tr_idx)
    hist = []
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = 0
        perm = tr_idx[np.random.choice(len(tr_idx), len(tr_idx), replace=True, p=sw)] if sw is not None else tr_idx[np.random.permutation(len(tr_idx))]
        frozen = ep < LPFT_EPOCHS
        for p_ in model.parameters():
            if id(p_) not in head: p_.requires_grad_(not frozen)
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            b = torch.from_numpy(perm[i:i+BATCH])
            x = norm(gpu_augment(to_input(TR[b]))).contiguous(memory_format=torch.channels_last); yb = yt[b].to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16): loss = crit(model(x), yb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_([p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model); tot += loss.item()
        if ep == 0: print(f"fold {k}: peak GPU mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
        pe = predict(ema.module, TR[torch.from_numpy(es_idx)])              # inner holdout: used for early stopping only
        nll = float(-np.log(np.clip(pe[np.arange(len(es_idx)), y[es_idx]], 1e-7, 1)).mean()); f1e = f1_score(y[es_idx], pe.argmax(1), average="macro")
        hist.append((ep + 1, round(nll, 4), round(f1e, 4)))
        if nll < best_nll - 1e-4: best_nll, best_ep, bad = nll, ep + 1, 0; best_state = {k_: v.detach().clone() for k_, v in ema.module.state_dict().items()}
        else: bad += 1
        print(f"fold {k} ep {ep+1}/{EPOCHS} loss {tot/max(1, len(perm)//BATCH):.4f} holdout(EMA) nll {nll:.4f} macroF1 {f1e:.4f} best_ep {best_ep} ({time.time()-t0:.0f}s)", flush=True)
        if ep + 1 >= ES_MIN_EP and bad >= ES_PATIENCE: print(f"early stop fold {k} at ep {ep+1}, best ep {best_ep}", flush=True); break
    model = ema.module; model.load_state_dict(best_state)                    # best EMA by inner-holdout NLL (val fold untouched -> honest OOF)
    hist.append(("best_ep", best_ep))
    pv = predict_views(model, TR[torch.from_numpy(va_idx)]); pt = predict_views(model, TE)
    del model, opt, ema; gc.collect(); torch.cuda.empty_cache()
    return pv, pt, hist


# ============================================================== FULL-TRAIN refit (replaces the fold loop)
SEEDS = [int(s) for s in os.environ.get("T4_SEEDS", "42").split(",")]
ACCUM = PR.get("accum", 1)                                     # gradient accumulation (LayerNorm model -> same as a bigger batch)
FULL_EPOCHS = int(os.environ.get("T4_FULL_EPOCHS", "14"))
RECIPE = os.environ.get("T4_RECIPE", "v3")                     # "v3" = sub_v3_5f augmentation + class-weighted CE (ls .1), "v4" = v4 recipe
LOWQ_AUG = os.environ.get("T4_LOWQ_AUG", "0") == "1"           # JPEG-like down/up-sampling aug; OFF = no augmentation chosen from test EDA
def train_full(seed):
    seed_all(seed); tr_idx = np.arange(len(y))
    model, _ = create_model(); model = model.to(DEVICE).to(memory_format=torch.channels_last)
    if PR.get("grad_ckpt"): model.set_grad_checkpointing(True)
    head = set(id(p) for p in model.get_classifier().parameters())
    opt = timm.optim.create_optimizer_v2(model, opt="adamw", lr=LR / LLRD, weight_decay=WD, layer_decay=LLRD)
    for g in opt.param_groups:
        g["lr"] = g["lr"] * g.get("lr_scale", 1.0)
        if any(id(p_) in head for p_ in g["params"]): g["lr"] = LR * 10
    spe = len(tr_idx) // (BATCH * ACCUM); steps = FULL_EPOCHS * spe; warm = max(1, int(0.05 * steps)); lp = LPFT_EPOCHS * spe   # optimizer steps
    def lr_lambda(s):
        if s < lp: return min(1.0, (s + 1) / warm)
        if s < lp + spe: return 0.05 + 0.95 * (s - lp + 1) / spe
        t = (s - lp - spe) / max(1, steps - lp - spe); return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda); scaler = torch.amp.GradScaler("cuda")
    if RECIPE == "v3":                                         # v3: plain shuffle + class-weighted CE, label smoothing .1
        cnt = np.bincount(y, minlength=K); crit = nn.CrossEntropyLoss(weight=torch.tensor(len(y) / (K * cnt), dtype=torch.float32, device=DEVICE), label_smoothing=0.1); sw = None
    else: crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH); sw = sample_weights(tr_idx)
    ema = timm.utils.ModelEmaV3(model, decay=EMA_DECAY); yt = torch.from_numpy(y); hist = []
    for ep in range(FULL_EPOCHS):
        model.train(); t0 = time.time(); tot = 0
        perm = tr_idx[np.random.choice(len(tr_idx), len(tr_idx), replace=True, p=sw)] if sw is not None else tr_idx[np.random.permutation(len(tr_idx))]
        frozen = ep < LPFT_EPOCHS
        for p_ in model.parameters():
            if id(p_) not in head: p_.requires_grad_(not frozen)
        opt.zero_grad(set_to_none=True)
        for j, i in enumerate(range(0, len(perm) - BATCH + 1, BATCH)):
            b = torch.from_numpy(perm[i:i+BATCH])
            x = norm((gpu_augment_v3 if RECIPE == "v3" else gpu_augment)(to_input(TR[b]))).contiguous(memory_format=torch.channels_last); yb = yt[b].to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16): loss = crit(model(x), yb)
            scaler.scale(loss / ACCUM).backward(); tot += loss.item()
            if (j + 1) % ACCUM == 0:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_([p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); sched.step(); ema.update(model); opt.zero_grad(set_to_none=True)
        if ep == 0: print(f"seed {seed}: peak GPU mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
        hist.append((ep + 1, round(tot / max(1, len(perm) // BATCH), 4), round(time.time() - t0)))
        if ep == 0: print(f"seed {seed}: ~{hist[-1][2] * FULL_EPOCHS / 3600:.1f} h for {FULL_EPOCHS} epochs", flush=True)
        print(f"seed {seed} ep {ep+1}/{FULL_EPOCHS} loss {hist[-1][1]} ({hist[-1][2]}s)", flush=True)
    pt = predict_views(ema.module, TE)
    del model, opt, ema; gc.collect(); torch.cuda.empty_cache()
    return pt, hist

os.makedirs(f"{OUT}/preds", exist_ok=True); times = []
for seed in SEEDS:
    el_h = (time.time() - T_START) / 3600; est = np.mean(times) / 3600 if times else 0
    if el_h + est > TIME_LIMIT_H: print(f"time guard: {el_h:.1f} h used, skipping seed {seed}"); break
    t0 = time.time(); pt, hist = train_full(seed); times.append(time.time() - t0); tag = f"v4f_{BACKBONE}_{RECIPE}aug_s{seed}"
    for v in TTA_VIEWS: np.save(f"{OUT}/preds/{tag}_test_full_{v}.npy", pt[v])
    P = np.mean([pt[v] for v in ("plain", "hflip", "up")], 0)
    json.dump({"model": MODEL_NAME, "config": PR, "seed": seed, "epochs": FULL_EPOCHS, "lowq_aug": LOWQ_AUG, "recipe": RECIPE, "hist": hist, "minutes": round(times[-1] / 60, 1),
               "test_share": dict(zip(CLASSES, (np.bincount(P.argmax(1), minlength=K) / len(P)).round(3).tolist()))},
              open(f"{OUT}/preds/{tag}_log.json", "w"), indent=1)
    print(f"== {tag} done in {times[-1]/60:.0f} min | elapsed {(time.time()-T_START)/60:.0f} min", flush=True)
print("wrote preds/ - nothing submitted"); raise SystemExit(0)
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
os.makedirs(f"{OUT}/preds", exist_ok=True)
def oof_path(v): return f"{OUT}/preds/{RUN_TAG}_oof_{v}.npy"
oof = {v: (np.load(oof_path(v)) if os.path.exists(oof_path(v)) else np.full((len(y), K), np.nan)) for v in TTA_VIEWS}
log = {"backbone": BACKBONE, "model": MODEL_NAME, "config": PR, "folds": {}}
fold_times = []
for k in RUN_FOLDS:
    el_h = (time.time() - T_START) / 3600; est = np.mean(fold_times) / 3600 if fold_times else 0
    if el_h + est > TIME_LIMIT_H: print(f"time guard: {el_h:.1f} h used, skipping folds from {k} (re-run with T4_FOLDS=...)"); break
    t0 = time.time(); tr_idx, va_idx = folds[k]
    pv, pt, hist = train_fold(k, tr_idx, va_idx); fold_times.append(time.time() - t0)
    for v in TTA_VIEWS:
        oof[v][va_idx] = pv[v]; np.save(oof_path(v), oof[v]); np.save(f"{OUT}/preds/{RUN_TAG}_test_fold{k}_{v}.npy", pt[v])
    pdef = np.mean([pv[v] for v in TTA_DEFAULT], 0)
    np.save(f"{OUT}/preds/{RUN_TAG}_test_fold{k}.npy", np.mean([pt[v] for v in TTA_DEFAULT], 0))
    log["folds"][k] = {"hist": hist, **{v: round(f1_score(y[va_idx], pv[v].argmax(1), average="macro"), 4) for v in TTA_VIEWS},
                       "tta_default": round(f1_score(y[va_idx], pdef.argmax(1), average="macro"), 4), "minutes": round(fold_times[-1] / 60, 1)}
    print(f"== fold {k}: {log['folds'][k]} | elapsed {(time.time()-T_START)/60:.0f} min", flush=True)
    json.dump(log, open(f"{OUT}/preds/{RUN_TAG}_log.json", "w"), indent=1)

# ============================================================== summary over all folds present on disk
m = ~np.isnan(oof["plain"][:, 0])
OOF = np.mean([oof[v] for v in TTA_DEFAULT], 0); np.save(f"{OUT}/preds/{RUN_TAG}_oof.npy", OOF)
print(f"\nOOF rows available: {m.sum()}/{len(y)}")
for v in TTA_VIEWS: print(f"  OOF macroF1 view {v:6s}: {f1_score(y[m], oof[v][m].argmax(1), average='macro'):.4f}")
print(f"  OOF macroF1 TTA {TTA_DEFAULT}: {f1_score(y[m], OOF[m].argmax(1), average='macro'):.4f}   (v3_5f: 0.9889)")
print(classification_report(y[m], OOF[m].argmax(1), target_names=CLASSES, digits=4))
print(pd.DataFrame(confusion_matrix(y[m], OOF[m].argmax(1)), index=CLASSES, columns=CLASSES))
tf = [f"{OUT}/preds/{RUN_TAG}_test_fold{k}.npy" for k in range(N_FOLDS) if os.path.exists(f"{OUT}/preds/{RUN_TAG}_test_fold{k}.npy")]
if len(tf) == N_FOLDS:
    P = np.mean([np.load(f) for f in tf], 0); np.save(f"{OUT}/preds/task4_{RUN_TAG}_test_proba.npy", P)
    print("test predicted class share:", dict(zip(CLASSES, (np.bincount(P.argmax(1), minlength=K) / len(P)).round(3))))
    res = pd.DataFrame({"id_img": test["id"], "pred_num": P.argmax(1), "pred_str": [CLASSES[i] for i in P.argmax(1)]})
    if SAMPLE:
        ids = pd.read_csv(SAMPLE[0], sep=None, engine="python")["id"]; assert len(ids) == len(test); res.insert(0, "id", ids.values)
    res.to_csv(f"{OUT}/preds/task4_{RUN_TAG}.csv", index=False); print(res.head())
else:
    print(f"only {len(tf)}/{N_FOLDS} test folds present - run the missing folds (T4_FOLDS=...) then re-run this cell")
print("wrote preds/ - nothing submitted")
