# Builds t4/colab/task4_lora7b_colab.ipynb: LoRA fine-tune of DINOv3 ViT-7B truncated at 75% depth (block 30 of 40 = the
# frozen "l75" features, best frozen head: SGKF .9683). Blocks 0-19 frozen (no grad), LoRA r16 on every Linear of blocks 20-29,
# head = LayerNorm + Linear on [CLS, mean patch]. Train labels only, generic augmentation (crop/flip/colour jitter), no test use.
# Runs (setting RUNS): "val0" = train on SGKF scene folds != 0, score held-out scene fold 0 (compare: frozen l75 + FT scene CV);
#                      "full" = train on all train rows, predict test plain + hflip.
# Output MyDrive/cpe342_t4/lora7b/preds/: v4f_lora7b_l75_s42_test_full_{plain,hflip}.npy (+ val0 files, logs, test_ids.csv).
# Local: copy to t4/preds/v4f_lora7b/ and add FT key "lora7b": ("t4/preds/v4f_lora7b", "v4f_lora7b_l75_s42", ["plain", "hflip"]).
# usage: python t4/colab/make_colab_lora7b.py
import base64, json, zlib, numpy as np
FOLD = np.load("t4/research/groupcv/sgkf_sig8_fold.npy").astype(np.int8)
FOLD_B64 = base64.b64encode(zlib.compress(FOLD.tobytes(), 9)).decode()
C = []
md = lambda s: C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: C.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# Task 4: LoRA fine-tune of DINOv3 ViT-7B (cut at 75% depth) — Colab A100

Train labels only; nothing is submitted.

**Before you start**
1. Runtime → **A100 GPU** (80 GB best; 40 GB works with `BS = 16`) + **High-RAM**.
2. 🔑 Secrets: `HF_TOKEN` (DINOv3 licence accepted), Notebook access on.
3. `task4_data.zip` in `MyDrive/cpe342_t4/`.
4. **Run all.** `RUNS` is a queue run in order (the 7B is loaded once). Entry = `<run>[@c<CUT>f<LORA_FROM>]`:
   `full_s7` = recipe default (cut 30, LoRA from 20), seed 7; `val0@c40f28` = cut 40 (all blocks), LoRA on 28-39, scored on scene fold 0.
   Time on A100: c30f20 val ~55 min / full ~65 min; c40f28 ~65 / ~85; c24f12 ~50 / ~60. Each epoch is checkpointed to Drive;
   after a disconnect **Run all** again resumes; finished runs are skipped.
   Cell 6 (`TTA`) scores the l75 LoRA at several image sizes on scene fold 0 and writes test predictions
   at those sizes (~50 min). Every TTA file is skipped if it already exists.
   Cell 7 (`SOUP = True`): model soup. Fold-0 check (~15 min), then one full soup per `SOUP_FULL` entry (~25 min each).
   Queue now: val0 + full_s42 of the logit-adjusted l75 (`@c30f20la`, ~2 h); fold-0 checks: refit-only controls (single l75,
   single c30f10), soup2 with a 2-view head, soup2la, soup3la (~15 min each); full soups soup2 (2-view head only), soup2la, soup3la
   (~35 min each). ~5 h in total; every finished piece is saved, so after a disconnect Run all continues.
   Cell 8 (`SOUP_TTA`, off): the soups at the extra TTA sizes, fold-0 check first. Finished soups / files are skipped,
   so Run all goes straight to the new part.

**Output**: `MyDrive/cpe342_t4/lora7b/preds/` → send the whole `preds` folder back (zip it, it is small).
""")

code(f"""
# ===== 1. settings
MODEL = "vit_7b_patch16_dinov3.lvd1689m"
CUT = 30            # keep blocks 0..29 (= l75 features), drop the last 10 blocks and the final norm
LORA_FROM = 20      # LoRA on blocks 20..29, blocks 0..19 stay frozen
RANK, ALPHA = 16, 32
SIZE = (224, 416)   # (H, W), patch 16
EPOCHS = 3
BS = 32             # 16 on a 40 GB A100
LR_LORA, LR_HEAD = 1e-4, 1e-3
WD_LORA, WD_HEAD = 0.01, 0.05
SMOOTH = 0.1
TAU = 1.0           # logit-adjustment strength for "la" runs (train class counts only)
SEED = 42
RUNS = ["val0@c30f10", "full_s42@c30f10", "val0@c30f20la", "full_s42@c30f20la"]   # c30f10 runs are done (skipped); kept so LoRA layers
# exist on blocks 10-29 for the soups. "la" = logit-adjusted loss (Menon et al. 2021): CE(logits + TAU * log train_prior), no class
# weights; the raw softmax is then already balanced (uniform target prior), so no post-hoc prior is applied to it.
# "valK" = train on scene folds != K, score fold K; "full" / "full_sN" = all train rows, seed N (default SEED), predict test;
# optional "@cXfY[la]" = keep blocks 0..X-1, LoRA on blocks Y..X-1 (default CUT / LORA_FROM above), "la" = logit-adjusted loss
TTA = False         # cell 6: multi-scale TTA of the l75 LoRA (fold-0 check from the val0 checkpoint + test for TTA_SEEDS)
TTA_SIZES = [(224, 416), (192, 352), (256, 480), (288, 528)]   # (H, W), multiples of 16; first = training size
TTA_SEEDS = [42]    # full-train l75 runs whose *_weights.pt are in the preds folder; add 7 if time allows (~40 min each)
SOUP = True         # cell 7: weight-space soup of LoRA runs with cut 30 (exact average of B@A, head refit on soup features)
# fold-0 check uses the val0 checkpoints of l75 (LoRA 20-29) and c30f10 (LoRA 10-29); full soups use *_weights.pt in the preds folder
SOUP_FULL = {{"soup2": [("l75_s42", 20), ("c30f10r16e3_s42", 10)],                     # same pair as the fold-0 check
             "soup2la": [("l75la_s42", 20), ("c30f10r16e3_s42", 10)],                 # l75 replaced by the logit-adjusted run
             "soup3la": [("l75_s42", 20), ("c30f10r16e3_s42", 10), ("l75la_s42", 20)]}}
SOUP_VAL = {{"single_l75": [("l75", 20)],                                  # control: head refit WITHOUT a soup (is the gain the refit?)
            "single_c30f10": [("c30f10r16e3", 10)],
            "soup2b": [("l75", 20), ("c30f10r16e3", 10)],                  # soup2 again, now also with the 2-view head refit
            "soup2la": [("l75la", 20), ("c30f10r16e3", 10)],
            "soup3la": [("l75", 20), ("c30f10r16e3", 10), ("l75la", 20)]}}
HEAD_EPOCHS, HEAD_LR = 5, 5e-4   # head refit on frozen soup features, warm-started from the averaged head
SOUP_TTA = []      # was ["soup2", "soup3"]; multi-scale TTA did not help on the LB (v84a); cell 8 = soup multi-scale TTA
DRIVE_DIR = "/content/drive/MyDrive/cpe342_t4"
FOLD_B64 = "{FOLD_B64}"   # SGKF scene folds (t4/research/groupcv/sgkf_sig8_fold.npy), train.csv row order
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
OUT = f"{DRIVE_DIR}/lora7b/preds"; CK = f"{DRIVE_DIR}/lora7b/ckpt"; os.makedirs(OUT, exist_ok=True); os.makedirs(CK, exist_ok=True)
print("timm", timm.__version__, "| out", OUT)
""")

code("""
# ===== 3. data -> local SSD -> uint8 arrays (240x448 cache), labels, scene folds
import glob, zipfile, time, base64, zlib, json, math, numpy as np, pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
DATA = "/content/data"; os.makedirs(DATA, exist_ok=True)
if not glob.glob(f"{DATA}/**/train.csv", recursive=True):
    z = f"{DRIVE_DIR}/task4_data.zip"
    if not os.path.exists(z): raise SetupError(f"upload task4_data.zip to {z}")
    zipfile.ZipFile(z).extractall(DATA)
ROOT = [os.path.dirname(p) for p in glob.glob(f"{DATA}/**/train.csv", recursive=True)][0]
train = pd.read_csv(f"{ROOT}/train.csv"); test = pd.read_csv(f"{ROOT}/test.csv")
CLASSES = sorted(train["class"].unique()); K = len(CLASSES)
Y = torch.tensor(train["class"].map({c: i for i, c in enumerate(CLASSES)}).values)
FOLD = np.frombuffer(zlib.decompress(base64.b64decode(FOLD_B64)), dtype=np.int8).copy()
assert len(FOLD) == len(train) == 31546 and list(train["id"].astype(str)[:3]) == ["su7tkav", "shlk51f", "sbdqvs0"], "train.csv order changed"
pd.DataFrame({"id_img": test["id"]}).to_csv(f"{OUT}/test_ids.csv", index=False)   # row order of every test file
def load_all(folder, names):
    read = lambda i: np.asarray(Image.open(f"{folder}/{i}.jpg").convert("RGB").resize((448, 240)))
    with ThreadPoolExecutor(32) as ex: return torch.from_numpy(np.stack(list(ex.map(read, names, chunksize=64))))
t0 = time.time()
X = {"train": load_all(f"{ROOT}/train", train["id"].astype(str).values), "test": load_all(f"{ROOT}/test", test["id"].astype(str).values)}
print({k: tuple(v.shape) for k, v in X.items()}, "classes", CLASSES, "folds", np.bincount(FOLD - FOLD.min()), f"{time.time()-t0:.0f}s")
""")

code("""
# ===== 4. model: 7B cut at block CUT, frozen bf16 base + LoRA (fp32) on blocks LORA_FROM..CUT-1, fp32 head
import torch.nn as nn, torch.nn.functional as F, gc
from torchvision.ops import roi_align
t0 = time.time()
model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=SIZE, dynamic_img_size=True)
cfg = timm.data.resolve_model_data_config(model)
def parse(r):   # "full_s7@c40f28" -> ("full_s7", 40, 28, False); "val0@c30f20la" -> ("val0", 30, 20, True)
    n, _, rc = r.partition("@"); la = rc.endswith("la"); rc = rc[:-2] if la else rc
    c, f = (CUT, LORA_FROM) if not rc else map(int, rc.lstrip("c").split("f"))
    return n, c, f, la
SPECS = [parse(r) for r in RUNS]; NB = len(model.blocks)
assert all(0 <= f < c <= NB for _, c, f, _ in SPECS), f"bad @cXfY (model has {NB} blocks)"
MAXCUT = max(c for _, c, _, _ in SPECS); MINF = min(f for _, _, f, _ in SPECS)
ALL = list(model.blocks)[:MAXCUT]; model.blocks = nn.ModuleList(ALL); model.norm = nn.Identity()
model = model.to(torch.bfloat16).to("cuda").eval()
for p in model.parameters(): p.requires_grad_(False)
gc.collect(); torch.cuda.empty_cache()

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__(); self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, device="cuda")); self.B = nn.Parameter(torch.zeros(base.out_features, r, device="cuda"))
        self.reset()
    def reset(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)); nn.init.zeros_(self.B)
    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale
LORA = {i: [] for i in range(MINF, MAXCUT)}   # block index -> its LoRA layers
for i in range(MINF, MAXCUT):
    blk = ALL[i]
    for name, mod in list(blk.named_modules()):
        for cn, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                l = LoRALinear(child, RANK, ALPHA); setattr(mod, cn, l); LORA[i].append(l)
from timm.layers.config import set_reentrant_ckpt
set_reentrant_ckpt(False)   # non-reentrant: block LORA_FROM gets a no-grad input, LoRA grads must still flow
model.set_grad_checkpointing(True)
NP = model.num_prefix_tokens; D = model.embed_dim
MEAN = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1); STD = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
head = nn.Sequential(nn.LayerNorm(2 * D), nn.Dropout(0.1), nn.Linear(2 * D, K)).cuda()
def activate(cut, f):   # use blocks 0..cut-1, train LoRA of blocks f..cut-1; all other LoRA layers are zero (B = 0) and frozen
    model.blocks = nn.ModuleList(ALL[:cut])
    for ls in LORA.values():
        for l in ls: l.reset(); l.A.requires_grad_(False); l.B.requires_grad_(False)
    act = [l for i in range(f, cut) for l in LORA[i]]
    for l in act: l.A.requires_grad_(True); l.B.requires_grad_(True)
    return act
print(f"{MAXCUT} blocks kept | LoRA on blocks {MINF}-{MAXCUT-1}, {sum(len(v) for v in LORA.values())} layers | "
      f"prefix {NP} | GPU {torch.cuda.memory_allocated()/1e9:.1f} GB | {time.time()-t0:.0f}s")

def embed(x):   # x: float (B,3,H,W) in [0,1] -> [CLS, mean patch] fp32
    t = model.forward_features(((x - MEAN) / STD).to(torch.bfloat16))
    return torch.cat([t[:, 0], t[:, NP:].mean(1)], 1).float()

def augment(u8):   # u8: (B,240,448,3) uint8 on GPU -> random resized crop + flip + colour jitter, (B,3,H,W) float
    x = u8.permute(0, 3, 1, 2).float().div_(255); B, _, h, w = x.shape
    area = torch.empty(B, device="cuda").uniform_(0.5, 1.0); ar = torch.exp(torch.empty(B, device="cuda").uniform_(-0.15, 0.15))
    cw = (w * torch.sqrt(area * ar)).clamp(max=w); ch = (h * torch.sqrt(area / ar)).clamp(max=h)
    x0 = torch.rand(B, device="cuda") * (w - cw); y0 = torch.rand(B, device="cuda") * (h - ch)
    boxes = torch.stack([torch.arange(B, device="cuda").float(), x0, y0, x0 + cw, y0 + ch], 1)
    x = roi_align(x, boxes, output_size=SIZE, spatial_scale=1.0, sampling_ratio=2, aligned=True)
    flip = torch.rand(B, device="cuda") < 0.5; x[flip] = x[flip].flip(3)
    j = lambda lo, hi: torch.empty(B, 1, 1, 1, device="cuda").uniform_(lo, hi)
    x = x * j(0.8, 1.2)                                                     # brightness
    m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * j(0.8, 1.2) + m      # contrast
    g = x.mean(1, keepdim=True); x = (x - g) * j(0.8, 1.2) + g              # saturation
    return x.clamp_(0, 1)

def plain(u8, view):
    x = u8.permute(0, 3, 1, 2).float().div_(255)
    x = F.interpolate(x, size=SIZE, mode="bicubic", align_corners=False, antialias=True).clamp_(0, 1)
    return x.flip(3) if view == "hflip" else x
""")

code("""
# ===== 5. train / predict
@torch.no_grad()
def predict(Xs, view, bs=64):
    model.eval(); head.eval(); out = []
    for i in range(0, len(Xs), bs):
        out.append(torch.softmax(head(embed(plain(Xs[i:i+bs].cuda(non_blocking=True), view))), 1).cpu())
    return torch.cat(out).numpy()

def run(name, tr_idx, seed, cut, f, var, la=False):
    torch.manual_seed(seed); np.random.seed(seed)
    act = activate(cut, f); lora_params = [p for l in act for p in (l.A, l.B)]
    print(f"[{name}] blocks 0-{cut-1}, LoRA {f}-{cut-1}: {len(act)} layers, {sum(p.numel() for p in lora_params)/1e6:.1f}M params", flush=True)
    for m in head.modules():
        if hasattr(m, "reset_parameters"): m.reset_parameters()
    opt = torch.optim.AdamW([{"params": lora_params, "lr": LR_LORA, "weight_decay": WD_LORA},
                             {"params": head.parameters(), "lr": LR_HEAD, "weight_decay": WD_HEAD}])
    spe = len(tr_idx) // BS; total = spe * EPOCHS; warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    cnt = torch.bincount(Y[tr_idx], minlength=K).float(); cw = 1 / torch.sqrt(cnt + 1); cw = (cw / cw.mean()).cuda()
    if la:   # logit adjustment: train on logits + TAU * log(prior); predict with the raw logits (balanced)
        lp = (TAU * torch.log(cnt / cnt.sum())).cuda(); ce = nn.CrossEntropyLoss(label_smoothing=SMOOTH); lossf = lambda z, t: ce(z + lp, t)
        print(f"[{name}] logit-adjusted loss, TAU {TAU}, no class weights", flush=True)
    else: lossf = nn.CrossEntropyLoss(weight=cw, label_smoothing=SMOOTH)
    ck = f"{CK}/{var}_{name}.pt"; ep0 = 0
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cuda", weights_only=False)   # own checkpoint
        assert len(s["lora"]) == len(act)
        for l, (a, b) in zip(act, s["lora"]): l.A.data.copy_(a); l.B.data.copy_(b)
        head.load_state_dict(s["head"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"]); ep0 = s["epoch"]
        print(f"[{name}] resumed after epoch {ep0}")
    log = {"name": name, "epochs": []}
    for ep in range(ep0, EPOCHS):
        model.train(); head.train(); g = torch.Generator().manual_seed(seed + ep)
        perm = torch.from_numpy(tr_idx)[torch.randperm(len(tr_idx), generator=g)]
        t0 = time.time(); run_loss = 0.0
        for st in range(spe):
            b = perm[st * BS:(st + 1) * BS]
            x = augment(X["train"][b].cuda(non_blocking=True)); yb = Y[b].cuda()
            loss = lossf(head(embed(x)), yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + list(head.parameters()), 1.0); opt.step(); sched.step()
            run_loss += loss.item()
            if st % 100 == 0 or st == spe - 1:
                el = time.time() - t0; eta = el / (st + 1) * (spe - st - 1)
                print(f"[{name}] ep {ep+1}/{EPOCHS} step {st+1}/{spe} loss {run_loss/(st+1):.4f} | {el/60:.1f} min, epoch ETA {eta/60:.1f} min"
                      f" | GPU peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
        log["epochs"].append({"epoch": ep + 1, "loss": run_loss / spe, "min": round((time.time() - t0) / 60, 1)})
        torch.save({"lora": [(l.A.data, l.B.data) for l in act], "head": head.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep + 1}, ck + ".tmp"); os.replace(ck + ".tmp", ck)
    return log, act

from sklearn.metrics import f1_score
for name, cut, f, la in SPECS:
    VAR = ("l75" if (cut, f, RANK, EPOCHS) == (30, 20, 16, 3) else f"c{cut}f{f}r{RANK}e{EPOCHS}") + ("la" if la else "")   # recipe id in file names
    if name.startswith("full"):
        seed = int(name.split("_s")[1]) if "_s" in name else SEED; tag = f"v4f_lora7b_{VAR}_s{seed}"; tr_idx = np.arange(len(FOLD))
    elif name.startswith("val"):
        vk = int(name[3:]); assert vk in set(FOLD.tolist()), f"no scene fold {vk}"
        seed = SEED; tag = f"lora7b_{VAR}_val{vk}"; tr_idx = np.where(FOLD != vk)[0]
    else: raise SetupError(f"unknown run name {name!r}: use valK, full or full_sN")
    if os.path.exists(f"{OUT}/{tag}_log.json"): print("skip (done)", name); continue
    print(f"=== {name}: seed {seed}, {len(tr_idx)} train rows, tag {tag}", flush=True)
    log, act = run(name, tr_idx, seed, cut, f, VAR, la)
    if name.startswith("val"):
        va = np.where(FOLD == vk)[0]; P = {v: predict(X["train"][va], v) for v in ("plain", "hflip")}
        for v, p in P.items(): np.save(f"{OUT}/{tag}_oof_{v}.npy", p)
        np.save(f"{OUT}/{tag}_val_idx.npy", va); yv = Y[va].numpy()
        log["val_f1"] = {v: round(f1_score(yv, p.argmax(1), average="macro"), 5) for v, p in P.items()}
        log["val_f1"]["tta"] = round(f1_score(yv, (P["plain"] + P["hflip"]).argmax(1), average="macro"), 5)
        print(f"[{name}] held-out scene fold {vk} ({len(va)} rows) macro-F1 {log['val_f1']}", flush=True)
    else:
        for v in ("plain", "hflip"): np.save(f"{OUT}/{tag}_test_full_{v}.npy", predict(X["test"], v))
        torch.save({"lora": [(l.A.data.cpu(), l.B.data.cpu()) for l in act], "head": head.state_dict()}, f"{OUT}/{tag}_weights.pt")
    json.dump(log, open(f"{OUT}/{tag}_log.json", "w"), indent=1); print("saved", tag, flush=True)
print("done - nothing submitted")
""")

code("""
# ===== 6. multi-scale TTA for the l75 LoRA (cut 30, LoRA 20-29): fold-0 check, then test predictions per size
@torch.no_grad()
def predict_sz(Xs, view, size, bs=32):
    model.eval(); head.eval(); out = []
    for i in range(0, len(Xs), bs):
        x = Xs[i:i+bs].cuda(non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        x = F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True).clamp_(0, 1)
        out.append(torch.softmax(head(embed(x.flip(3) if view == "hflip" else x)), 1).cpu())
    return torch.cat(out).numpy()
def load_l75(path):   # LoRA order = blocks 20..29 in named_modules order, as saved by every l75 run
    act = activate(30, 20); s = torch.load(path, map_location="cuda", weights_only=False)
    assert len(s["lora"]) == len(act), (len(s["lora"]), len(act))
    for l, (a, b) in zip(act, s["lora"]): l.A.data.copy_(a); l.B.data.copy_(b)
    head.load_state_dict(s["head"]); print("loaded", path, flush=True)
if TTA:
    assert MINF <= 20 and MAXCUT >= 30, "TTA needs LoRA layers on blocks 20-29: keep RUNS within cut >= 30, LoRA from <= 20"
    ck0 = next((q for q in (f"{CK}/l75_val0.pt", f"{CK}/val0.pt") if os.path.exists(q)), None)
    if ck0 is None: print("no l75 val0 checkpoint in", CK, "-> fold-0 TTA check skipped")
    else:
        load_l75(ck0); va = np.where(FOLD == 0)[0]; yv = Y[va].numpy(); Xv = X["train"][va]; got = {}
        for sz in TTA_SIZES:
            fn = f"{OUT}/lora7b_l75_val0_tta_{sz[0]}x{sz[1]}.npy"
            if not os.path.exists(fn): np.save(fn, (predict_sz(Xv, "plain", sz) + predict_sz(Xv, "hflip", sz)) / 2)
            got[sz] = np.load(fn); pr = got[sz].argmax(1)
            print(f"[tta fold0] {sz}: macro-F1 {f1_score(yv, pr, average='macro'):.5f}, errors {(pr != yv).sum()}", flush=True)
        pr = np.mean([np.log(np.clip(v, 1e-9, 1)) for v in got.values()], 0).argmax(1)
        print(f"[tta fold0] all sizes (geo-mean): macro-F1 {f1_score(yv, pr, average='macro'):.5f}, errors {(pr != yv).sum()}"
              "  (training size alone should match val0: .99115 / 42 errors)", flush=True)
    for sd in TTA_SEEDS:
        w = f"{OUT}/v4f_lora7b_l75_s{sd}_weights.pt"
        if not os.path.exists(w): print("missing", w, "-> skipped"); continue
        todo = [(sz, v) for sz in TTA_SIZES[1:] for v in ("plain", "hflip")
                if not os.path.exists(f"{OUT}/v4f_lora7b_l75_s{sd}_test_full_{v}_{sz[0]}x{sz[1]}.npy")]
        if not todo: print("seed", sd, "TTA done"); continue
        load_l75(w)
        for sz, v in todo:
            t0 = time.time(); np.save(f"{OUT}/v4f_lora7b_l75_s{sd}_test_full_{v}_{sz[0]}x{sz[1]}.npy", predict_sz(X["test"], v, sz))
            print(f"[tta test] seed {sd} {sz} {v}: {time.time()-t0:.0f}s", flush=True)
print("TTA cell done")
""")

code("""
# ===== 7. model soup: average the LoRA weight deltas of several cut-30 runs, then refit the head on the soup features
# B@A of each run is kept exactly: A_soup = [A_1; A_2; ...], B_soup = [B_1, B_2, ...] / n  ->  B_soup @ A_soup = mean_i(B_i @ A_i)
# (averaging A and B separately would be wrong: LoRA factors of different runs live in unrelated bases).
import copy
def soup_load(parts):   # parts: [(checkpoint path, lora_from)], all cut 30
    fmin = min(f for _, f in parts); act = activate(30, fmin); n = len(LORA[29]); w = 1 / len(parts)
    for l in act: l.A.requires_grad_(False); l.B.requires_grad_(False)
    got = {}; heads = []
    for path, f in parts:
        st = torch.load(path, map_location="cuda", weights_only=False); heads.append(st["head"])
        assert len(st["lora"]) == n * (30 - f), (path, len(st["lora"]), n * (30 - f))
        for k, (a, b) in enumerate(st["lora"]): got.setdefault((f + k // n, k % n), []).append((a.float(), b.float()))
    for i in range(fmin, 30):
        for j, l in enumerate(LORA[i]):
            pc = got.get((i, j))
            if pc: l.A = nn.Parameter(torch.cat([a for a, _ in pc], 0), requires_grad=False); l.B = nn.Parameter(torch.cat([b for _, b in pc], 1) * w, requires_grad=False)
    hs = copy.deepcopy(heads[0])
    for k in hs: hs[k] = sum(h[k].float() for h in heads) / len(heads)
    print("soup of", [os.path.basename(p) for p, _ in parts], flush=True); return hs
def soup_reset():   # back to plain rank-RANK LoRA layers (B = 0)
    for ls in LORA.values():
        for l in ls:
            l.A = nn.Parameter(torch.empty(RANK, l.base.in_features, device="cuda")); l.B = nn.Parameter(torch.zeros(l.base.out_features, RANK, device="cuda")); l.reset()
@torch.no_grad()
def embed_all(Xs, view, bs=64):
    model.eval(); return torch.cat([embed(plain(Xs[i:i+bs].cuda(non_blocking=True), view)).cpu() for i in range(0, len(Xs), bs)])
def new_head(state):
    h = nn.Sequential(nn.LayerNorm(2 * D), nn.Dropout(0.1), nn.Linear(2 * D, K)).cuda(); h.load_state_dict(state); return h.eval()
@torch.no_grad()
def head_proba(h, E): h.eval(); return torch.cat([torch.softmax(h(E[i:i+4096].cuda()), 1).cpu() for i in range(0, len(E), 4096)]).numpy()
def fit_head(E, y, state):   # same loss as LoRA training (class weights 1/sqrt(count), label smoothing), features fixed
    torch.manual_seed(SEED); h = new_head(state); h.train(); E = E.cuda(); y = y.cuda(); bs = 512
    cnt = torch.bincount(y, minlength=K).float(); cw = 1 / torch.sqrt(cnt + 1); cw = cw / cw.mean()
    lf = nn.CrossEntropyLoss(weight=cw, label_smoothing=SMOOTH); opt = torch.optim.AdamW(h.parameters(), lr=HEAD_LR, weight_decay=WD_HEAD)
    steps = HEAD_EPOCHS * math.ceil(len(E) / bs); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for ep in range(HEAD_EPOCHS):
        perm = torch.randperm(len(E), device="cuda")
        for i in range(0, len(E), bs):
            b = perm[i:i+bs]; opt.zero_grad(); lf(h(E[b]), y[b]).backward(); opt.step(); sch.step()
    return h.eval()
if SOUP:
    assert MINF <= 10 and MAXCUT >= 30, "soup needs LoRA layers on blocks 10-29: keep a run @c30f10 in RUNS"
    vck = lambda r: next((q for q in ((f"{CK}/l75_val0.pt", f"{CK}/val0.pt") if r == "l75" else (f"{CK}/{r}_val0.pt",)) if os.path.exists(q)), None)
    # heads per soup: avghead (averaged), refit (head refit on plain train features), refit2v (refit on plain + hflip features)
    for sname, spec in SOUP_VAL.items():
        tag = f"lora7b_{sname}_val0"; parts = [(vck(r), f) for r, f in spec]
        if any(p is None for p, _ in parts): print("val0 checkpoints missing for", sname, spec, "-> fold-0 check skipped"); continue
        if os.path.exists(f"{OUT}/{tag}_log.json"): print("skip (done)", tag); continue
        t0 = time.time(); hs = soup_load(parts)
        va = np.where(FOLD == 0)[0]; tr = np.where(FOLD != 0)[0]; yv = Y[va].numpy()
        Ev = {v: embed_all(X["train"][va], v) for v in ("plain", "hflip")}
        Ep = embed_all(X["train"][tr], "plain"); h1 = fit_head(Ep, Y[tr], hs)
        Eh = embed_all(X["train"][tr], "hflip"); h2 = fit_head(torch.cat([Ep, Eh]), torch.cat([Y[tr], Y[tr]]), hs); del Ep, Eh
        log = {"name": tag, "parts": spec}
        for hn, h in (("avghead", new_head(hs)), ("refit", h1), ("refit2v", h2)):
            P = {v: head_proba(h, Ev[v]) for v in Ev}; t = f"{tag}_{hn}"
            for v, p in P.items(): np.save(f"{OUT}/{t}_oof_{v}.npy", p)
            pr = (P["plain"] + P["hflip"]).argmax(1)
            log[hn] = {"f1": round(f1_score(yv, pr, average="macro"), 5), "errors": int((pr != yv).sum())}
            print(f"[soup fold0 {sname}] {hn}: macro-F1 {log[hn]['f1']}, errors {log[hn]['errors']}"
                  "  (compare: l75 .99115/42, c30f10 .99172/40, soup2 refit .9937/30)", flush=True)
        np.save(f"{OUT}/{tag}_val_idx.npy", va); json.dump(log, open(f"{OUT}/{tag}_log.json", "w"), indent=1)
        del Ev; gc.collect(); print(f"fold-0 soup {sname} {time.time()-t0:.0f}s", flush=True)
    for name, spec in SOUP_FULL.items():
        tag = f"v4f_lora7b_{name}_s42"; t2 = f"v4f_lora7b_{name}hv2_s42"; ta = f"v4f_lora7b_{name}avghead_s42"
        if os.path.exists(f"{OUT}/{t2}_test_full_hflip.npy"): print("skip (done)", name); continue
        parts = [(f"{OUT}/v4f_lora7b_{r}_weights.pt", f) for r, f in spec]
        miss = [p for p, _ in parts if not os.path.exists(p)]
        if miss: print("missing", miss, "->", name, "skipped"); continue
        t0 = time.time(); hs = soup_load(parts); have1 = os.path.exists(f"{OUT}/{tag}_test_full_hflip.npy")   # keep submitted files as they are
        Ep = embed_all(X["train"], "plain"); h1 = None if have1 else fit_head(Ep, Y, hs)
        Eh = embed_all(X["train"], "hflip"); h2 = fit_head(torch.cat([Ep, Eh]), torch.cat([Y, Y]), hs); del Ep, Eh
        h_avg = new_head(hs)
        for v in ("plain", "hflip"):
            E = embed_all(X["test"], v); np.save(f"{OUT}/{t2}_test_full_{v}.npy", head_proba(h2, E))
            if not have1:
                np.save(f"{OUT}/{tag}_test_full_{v}.npy", head_proba(h1, E)); np.save(f"{OUT}/{ta}_test_full_{v}.npy", head_proba(h_avg, E))
        if not have1: torch.save({"head_refit": h1.state_dict(), "head_avg": hs, "parts": spec}, f"{OUT}/{tag}_head.pt")
        torch.save({"head_refit2v": h2.state_dict(), "parts": spec}, f"{OUT}/{t2}_head.pt")
        print(f"saved {name} ({time.time()-t0:.0f}s)", flush=True)
    soup_reset()
print("soup cell done")
""")

code("""
# ===== 8. multi-scale TTA for the soups (refit head): fold-0 check, then test predictions at TTA_SIZES[1:]
if SOUP_TTA:
    va = np.where(FOLD == 0)[0]; yv = Y[va].numpy(); Xv = X["train"][va]; extra = TTA_SIZES[1:]
    todo = [sz for sz in extra if not os.path.exists(f"{OUT}/lora7b_soup2_val0_refit_tta_{sz[0]}x{sz[1]}.npy")]
    hv = f"{OUT}/lora7b_soup2_val0_head.pt"
    if todo:
        ck_l75 = next((q for q in (f"{CK}/l75_val0.pt", f"{CK}/val0.pt") if os.path.exists(q)), None)
        hs = soup_load([(ck_l75, 20), (f"{CK}/c30f10r16e3_val0.pt", 10)])
        if os.path.exists(hv): st = torch.load(hv, map_location="cuda", weights_only=False)
        else:
            tr = np.where(FOLD != 0)[0]; st = fit_head(embed_all(X["train"][tr], "plain"), Y[tr], hs).state_dict(); torch.save(st, hv)
        head.load_state_dict(st)
        for sz in todo:
            t0 = time.time(); np.save(f"{OUT}/lora7b_soup2_val0_refit_tta_{sz[0]}x{sz[1]}.npy", (predict_sz(Xv, "plain", sz) + predict_sz(Xv, "hflip", sz)) / 2)
            print(f"[soup tta fold0] {sz} {time.time()-t0:.0f}s", flush=True)
    lg = lambda p: np.log(np.clip(p, 1e-9, 1))
    got = {TTA_SIZES[0]: (np.load(f"{OUT}/lora7b_soup2_val0_refit_oof_plain.npy") + np.load(f"{OUT}/lora7b_soup2_val0_refit_oof_hflip.npy")) / 2}
    for sz in extra: got[sz] = np.load(f"{OUT}/lora7b_soup2_val0_refit_tta_{sz[0]}x{sz[1]}.npy")
    for sz, p in got.items(): pr = p.argmax(1); print(f"[soup tta fold0] {sz}: macro-F1 {f1_score(yv, pr, average='macro'):.5f}, errors {(pr != yv).sum()}", flush=True)
    pr = np.mean([lg(p) for p in got.values()], 0).argmax(1)
    print(f"[soup tta fold0] all sizes: macro-F1 {f1_score(yv, pr, average='macro'):.5f}, errors {(pr != yv).sum()}  (224x416 alone: .9937 / 30)", flush=True)
    for name in SOUP_TTA:
        tag = f"v4f_lora7b_{name}_s42"; hp = f"{OUT}/{tag}_head.pt"
        todo = [(sz, v) for sz in extra for v in ("plain", "hflip") if not os.path.exists(f"{OUT}/{tag}_test_full_{v}_{sz[0]}x{sz[1]}.npy")]
        if not todo: print(name, "TTA done"); continue
        if not os.path.exists(hp): print("missing", hp, "-> run cell 7 first"); continue
        hd = torch.load(hp, map_location="cuda", weights_only=False)
        soup_load([(f"{OUT}/v4f_lora7b_{r}_weights.pt", f) for r, f in hd["parts"]]); head.load_state_dict(hd["head_refit"])
        for sz, v in todo:
            t0 = time.time(); np.save(f"{OUT}/{tag}_test_full_{v}_{sz[0]}x{sz[1]}.npy", predict_sz(X["test"], v, sz))
            print(f"[soup tta test] {name} {sz} {v}: {time.time()-t0:.0f}s", flush=True)
    soup_reset()
print("soup TTA cell done")
""")

code("""
# ===== 9. zip the preds folder for sending back
import shutil; print(shutil.make_archive(f"{DRIVE_DIR}/lora7b_preds", "zip", OUT))
""")

nb = {"cells": C, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("t4/colab/task4_lora7b_colab.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote t4/colab/task4_lora7b_colab.ipynb | fold bytes", len(FOLD_B64))
