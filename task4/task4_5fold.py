# Task 4 — ConvNeXt-Tiny DINOv3 fine-tune, v3 config, FULL 5-fold CV + fold ensemble (macro F1)
# v3a fold0 = 0.9915 (best epoch 14) -> EPOCHS=14, all 5 folds; test = mean of 5 fold models' probabilities
# Needs: GPU T4, Internet ON, Kaggle secret HF_TOKEN, input dataset kiatisakkk/task4-image-classification (+ competition data for sample ids)
# NO SUBMIT - writes /kaggle/working only
import os, glob, time, gc, random, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
import timm.optim, timm.utils
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
warnings.filterwarnings("ignore")

RUN_FOLDS  = [0, 1, 2, 3, 4]
RUN_TAG    = "v3_5f"
BACKBONE   = "convnext_tiny"
LLRD, LPFT_EPOCHS = 0.8, 1
N_FOLDS, EPOCHS, BATCH = 5, 14, 64
LR, WD = 2e-4, 0.05
IMG_H, IMG_W = 192, 352
LABEL_SMOOTH, SEED, EMA_DECAY = 0.1, 42, 0.999
OUT, CACHE = "/kaggle/working", "/tmp/t4cache"

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN"); print("HF_TOKEN loaded")
except Exception as e:
    raise SystemExit(f"STOP: attach Kaggle secret HF_TOKEN (Add-ons -> Secrets) first ({type(e).__name__})")
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED); DEVICE = "cuda"
assert torch.cuda.is_available(), "turn on GPU T4"
torch.backends.cudnn.benchmark = True
print(torch.cuda.get_device_name(0), "| timm", timm.__version__, flush=True)

root = None
for _ in range(30):                                   # any input dir with train.csv, test.csv, train/, test/ (dataset or competition copy)
    hits = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train.csv", recursive=True)]
    hits = [h for h in hits if os.path.isdir(f"{h}/train") and os.path.isdir(f"{h}/test") and os.path.isfile(f"{h}/test.csv")]
    if hits: root = sorted(hits)[0]; break
    time.sleep(10)
assert root, "task4 image data not found"
SAMPLE = sorted(glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True))
train = pd.read_csv(f"{root}/train.csv"); test = pd.read_csv(f"{root}/test.csv")
CLASSES = sorted(train["class"].unique()); cid = {c: i for i, c in enumerate(CLASSES)}
y = train["class"].map(cid).values.astype(np.int64)
print(root, train.shape, test.shape, CLASSES, "| sample:", SAMPLE[:1], flush=True)

def load_all(folder, ids, name):
    os.makedirs(CACHE, exist_ok=True); p = f"{CACHE}/{name}.npy"
    if os.path.exists(p): return np.load(p)
    t0 = time.time(); read = lambda i: np.asarray(Image.open(f"{folder}/{i}.jpg").convert("RGB"))
    with ThreadPoolExecutor(32) as ex: arr = np.stack(list(ex.map(read, ids, chunksize=64)))
    np.save(p, arr); print(f"{name}: {arr.shape} in {time.time()-t0:.0f}s", flush=True); return arr
TR = torch.from_numpy(load_all(f"{root}/train", train["id"], "train"))
TE = torch.from_numpy(load_all(f"{root}/test", test["id"], "test"))

def pick_models():
    d3 = [m for m in timm.list_pretrained(f"{BACKBONE}*") if "dinov3" in m]
    return d3 + [f"{BACKBONE}.fb_in22k_ft_in1k", f"{BACKBONE}.fb_in22k"]
def create_model():
    for name in pick_models():
        try: return timm.create_model(name, pretrained=True, num_classes=len(CLASSES), drop_path_rate=0.1), name
        except Exception as e: print(f"skip {name}: {type(e).__name__}: {str(e)[:150]}")
    raise RuntimeError("cannot load model - check Internet / HF_TOKEN")
_m, MODEL_NAME = create_model()
cfg = timm.data.resolve_model_data_config(_m)
MEAN = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1); STD = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)
print("model:", MODEL_NAME, flush=True); del _m
assert "dinov3" in MODEL_NAME, "DINOv3 weights not loaded (check HF_TOKEN / Internet) - stopping instead of training a weaker model"

def to_input(xb_u8):
    x = xb_u8.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    return F.interpolate(x, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
def gpu_augment(x):
    B, _, H, W = x.shape; dev = x.device
    ang = torch.where(torch.rand(B, device=dev) < 0.7, (torch.rand(B, device=dev) * 60 - 30), torch.zeros(B, device=dev)) * torch.pi / 180
    s = torch.empty(B, device=dev).uniform_(0.85, 1.2); c, sn = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = c; theta[:, 0, 1] = -sn * H / W; theta[:, 1, 0] = sn * W / H; theta[:, 1, 1] = c
    theta[:, :, 2] = torch.empty(B, 2, device=dev).uniform_(-0.08, 0.08)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False)
    x = x * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.7, 1.3) + m
    g = x.mean(1, keepdim=True); x = (x - g) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.5, 1.6) + g
    perm = torch.rand(B, device=dev) < 0.25
    if perm.any():
        idx = torch.stack([torch.randperm(3, device=dev) for _ in range(int(perm.sum()))])
        x[perm] = torch.gather(x[perm], 1, idx[:, :, None, None].expand(-1, -1, H, W))
    inv = torch.rand(B, device=dev) < 0.1; x[inv] = 1 - x[inv]
    gr = torch.rand(B, device=dev) < 0.1; x[gr] = x[gr].mean(1, keepdim=True).expand(-1, 3, -1, -1)
    nz = (torch.rand(B, 1, 1, 1, device=dev) < 0.5).float() * torch.empty(B, 1, 1, 1, device=dev).uniform_(0, 0.08)
    return (x + torch.randn_like(x) * nz).clamp_(0, 1)
norm = lambda x: (x - MEAN) / STD

counts = np.bincount(y, minlength=len(CLASSES))
CLASS_W = torch.tensor(len(y) / (len(CLASSES) * counts), dtype=torch.float32, device=DEVICE)
@torch.no_grad()
def predict(model, X, bs=256):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(norm(to_input(X[i:i+bs])).contiguous(memory_format=torch.channels_last)).float().softmax(1).cpu())
    return torch.cat(out).numpy()

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
        t = (s - lp - spe) / max(1, steps - lp - spe); return 0.5 * (1 + np.cos(np.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda); scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss(weight=CLASS_W, label_smoothing=LABEL_SMOOTH)
    ema = timm.utils.ModelEmaV3(model, decay=EMA_DECAY); yt = torch.from_numpy(y)
    best, best_state, best_p = -1, None, None
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); perm = tr_idx[np.random.permutation(len(tr_idx))]; tot = 0
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
        p = predict(ema.module, TR[torch.from_numpy(va_idx)]); f1 = f1_score(y[va_idx], p.argmax(1), average="macro")
        print(f"fold {k} ep {ep+1}/{EPOCHS} loss {tot/(len(perm)//BATCH):.4f} val macroF1 {f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if f1 > best: best, best_p, best_state = f1, p, {k_: v.detach().clone() for k_, v in ema.module.state_dict().items()}
    model = ema.module; model.load_state_dict(best_state)
    pt = predict(model, TE)
    del model, opt, ema; gc.collect(); torch.cuda.empty_cache()
    return best, best_p, pt

folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
oof = np.full((len(y), len(CLASSES)), np.nan); test_probs = []; fold_f1 = {}
os.makedirs(f"{OUT}/preds", exist_ok=True)
t_all = time.time()
for k in RUN_FOLDS:
    tr_idx, va_idx = folds[k]
    f1k, p_va, p_te = train_fold(k, tr_idx, va_idx)
    oof[va_idx] = p_va; test_probs.append(p_te); fold_f1[k] = round(float(f1k), 4)
    np.save(f"{OUT}/preds/{RUN_TAG}_oof.npy", oof); np.save(f"{OUT}/preds/{RUN_TAG}_test_fold{k}.npy", p_te)
    print(f"== fold {k} best macroF1 {f1k:.4f} | elapsed {(time.time()-t_all)/60:.0f} min", flush=True)

m = ~np.isnan(oof[:, 0]); pr = oof[m].argmax(1); yt_ = y[m]
print(f"\nper-fold best macroF1: {fold_f1}")
print(f"OOF macro F1 (folds {RUN_FOLDS}) = {f1_score(yt_, pr, average='macro'):.4f}   (v3a fold0 alone 0.9915)")
print(classification_report(yt_, pr, target_names=CLASSES, digits=4))
print(pd.DataFrame(confusion_matrix(yt_, pr), index=CLASSES, columns=CLASSES))

P = np.mean(test_probs, 0); np.save(f"{OUT}/preds/task4_{RUN_TAG}_test_proba.npy", P)
agree = np.mean([(tp.argmax(1) == P.argmax(1)).mean() for tp in test_probs])
print(f"test: mean agreement of single folds with ensemble {agree:.4f}; predicted class share:",
      dict(zip(CLASSES, (np.bincount(P.argmax(1), minlength=len(CLASSES)) / len(P)).round(3))))
res = pd.DataFrame({"id_img": test["id"], "pred_num": P.argmax(1), "pred_str": [CLASSES[i] for i in P.argmax(1)]})
if SAMPLE:
    ids = pd.read_csv(SAMPLE[0], sep=None, engine="python")["id"]; assert len(ids) == len(test); res.insert(0, "id", ids.values)
res.to_csv(f"{OUT}/preds/task4_{RUN_TAG}.csv", index=False)
print(res.head()); print("wrote preds/ - nothing submitted")
