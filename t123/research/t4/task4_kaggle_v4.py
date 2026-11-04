# Task 4 v4 — Kaggle GPU training script (based on task4/task4_5fold.py = v3_5f, OOF macro-F1 0.9889)
# NOT run in the repo sandbox (no GPU / no images). Paste into a Kaggle notebook (GPU T4 x1 or P100, Internet ON).
# NO SUBMIT - writes /kaggle/working/preds only.
#
# Changes vs v3_5f (see t123/research/t4/REPORT.md):
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
BACKBONE   = os.environ.get("T4_BACKBONE", "cnxt_s_d3")       # cnxt_t_d3 | cnxt_s_d3 | eva02_s
RUN_FOLDS  = [int(f) for f in os.environ.get("T4_FOLDS", "0,1,2,3,4").split(",")]
PRESETS = {   # names tried in order; (H, W) must suit the stride (32 for ConvNeXt, 14 for EVA02)
    "cnxt_t_d3": dict(names=["convnext_tiny.dinov3_lvd1689m", "convnext_tiny.fb_in22k_ft_in1k"],
                      size=(256, 480), batch=48, lr=2e-4, epochs=20, llrd=0.8, dpr=0.1, need_hf=True),
    "cnxt_s_d3": dict(names=["convnext_small.dinov3_lvd1689m", "convnext_small.fb_in22k_ft_in1k"],
                      size=(224, 416), batch=32, lr=1.5e-4, epochs=18, llrd=0.8, dpr=0.2, need_hf=True),
    "eva02_s":   dict(names=["eva02_small_patch14_336.mim_in22k_ft_in1k"],
                      size=(224, 420), batch=32, lr=1e-4, epochs=18, llrd=0.75, dpr=0.1, need_hf=False),
}
PR = PRESETS[BACKBONE]
RUN_TAG    = f"v4_{BACKBONE}"
N_FOLDS, SEED = 5, 42                                          # SAME folds as v3 (StratifiedKFold(5, shuffle, 42)) -> OOFs blendable
IMG_H, IMG_W = PR["size"]; BATCH, LR, EPOCHS, LLRD = PR["batch"], PR["lr"], PR["epochs"], PR["llrd"]
WD, LPFT_EPOCHS, LABEL_SMOOTH, EMA_DECAY = 0.05, 1, 0.05, 0.9995
HFLIP_P    = 0.5                                               # set 0.0 if OOF of the hflip view is clearly worse
SAMPLER    = "sqrt"                                            # none | sqrt | balanced | prior
PRIOR_MIX  = None                                              # only for SAMPLER="prior": list of 5 floats (datacard order)
TTA_VIEWS  = {"plain": (False, 1.0), "hflip": (True, 1.0), "up": (False, 1.125)}
TTA_DEFAULT = ["plain", "hflip"]                               # views averaged into {TAG}_oof.npy / test_proba
TIME_LIMIT_H = 11.0                                            # stop starting new folds after this (Kaggle limit 12 h)
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
    lo = torch.rand(B, device=dev) < 0.25
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
    model, _ = create_model(); model = model.to(DEVICE).to(memory_format=torch.channels_last)
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
        if ep in (EPOCHS // 2, EPOCHS - 3, EPOCHS - 1) or ep == 0:         # monitoring only - NOT used for model selection
            p = predict(ema.module, TR[torch.from_numpy(va_idx)]); f1 = f1_score(y[va_idx], p.argmax(1), average="macro"); hist.append((ep + 1, round(f1, 4)))
        else: f1 = float("nan")
        print(f"fold {k} ep {ep+1}/{EPOCHS} loss {tot/max(1, len(perm)//BATCH):.4f} val(EMA) macroF1 {f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
    model = ema.module                                                       # final EMA weights (honest OOF)
    pv = predict_views(model, TR[torch.from_numpy(va_idx)]); pt = predict_views(model, TE)
    del model, opt, ema; gc.collect(); torch.cuda.empty_cache()
    return pv, pt, hist

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
    TFk = [np.load(f) for f in tf]; P = np.mean(TFk, 0); np.save(f"{OUT}/preds/task4_{RUN_TAG}_test_proba.npy", P)
    # label-free robustness DIAGNOSTIC (Generalization-Disagreement Equality, Jiang et al. 2022): mean pairwise
    # disagreement of fold models on test ~ single-model test error. v3_5f: 0.0866 (OOF error 0.011). Report it to
    # compare runs; do NOT tune hyper-parameters on it (it is computed on test images).
    dis = np.mean([(TFk[i].argmax(1) != TFk[j].argmax(1)).mean() for i in range(N_FOLDS) for j in range(i + 1, N_FOLDS)])
    print(f"test pairwise fold disagreement {dis:.4f} (v3_5f 0.0866) | all-5-agree {np.mean([(np.stack([t.argmax(1) for t in TFk]) == TFk[0].argmax(1)).all(0)]):.4f} (v3_5f 0.8337)")
    log["test_pairwise_fold_disagreement"] = round(float(dis), 4); json.dump(log, open(f"{OUT}/preds/{RUN_TAG}_log.json", "w"), indent=1)
    print("test predicted class share:", dict(zip(CLASSES, (np.bincount(P.argmax(1), minlength=K) / len(P)).round(3))))
    res = pd.DataFrame({"id_img": test["id"], "pred_num": P.argmax(1), "pred_str": [CLASSES[i] for i in P.argmax(1)]})
    if SAMPLE:
        ids = pd.read_csv(SAMPLE[0], sep=None, engine="python")["id"]; assert len(ids) == len(test); res.insert(0, "id", ids.values)
    res.to_csv(f"{OUT}/preds/task4_{RUN_TAG}.csv", index=False); print(res.head())
else:
    print(f"only {len(tf)}/{N_FOLDS} test folds present - run the missing folds (T4_FOLDS=...) then re-run this cell")
print("wrote preds/ - nothing submitted")
