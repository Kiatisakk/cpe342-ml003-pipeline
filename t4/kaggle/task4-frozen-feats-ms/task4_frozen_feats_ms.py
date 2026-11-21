# Task 4 - Kernel MS: MULTI-SCALE frozen features (no fine-tuning, nothing submitted). Strict-clean: views plain + hflip only,
# fixed scales chosen before seeing any result, no test statistics used for any choice.
# For every backbone and view it writes, in the format of Kernels B-E ([first token, mean patch], float16):
#   {bk}_s05  input about 112x208  (close to the native 120x230 resolution of the screenshots)
#   {bk}_s10  input about 224x432  (the resolution of every earlier frozen run; recomputed so all files come from one code path)
#   {bk}_s15  input about 336x640  (finer patch grid on small UI elements)
#   {bk}_l50, {bk}_l75  intermediate blocks at 1/2 and 3/4 depth, taken from the s10 pass (feature-pyramid levels)
# Files: <OUT>/{tag}_{split}_{view}.npy + train_names.npy / test_names.npy + log.json. Resumable: finished files are skipped,
# so after a timeout or a disconnect simply run it again (on Colab OUT points to Drive).
# Env: T4_FEATS   backbones in priority order, default "vithp,dinov2g,radio3,radio" (vithp is gated: HF_TOKEN from Kaggle
#                 Secrets / Colab secrets / env; without it vithp is skipped and the run goes on; "radio3g" optional)
# Estimated T4 time (3 scales + layers, from earlier single-scale logs x3.8; s15 not measured yet): vithp 7.4 h, dinov2g 9.9 h,
#   radio3 3.7 h, radio 4.8 h. Plan: session A "vithp", session B "dinov2g", session C "radio3,radio". /kaggle/working keeps
#   20 GB: about 3 / 3.5 / 4.7 / 3.8 GB of features per backbone.
#      T4_SCALES  default "s05,s10,s15";  T4_OUT, T4_DATA_ROOT;  T4_TIME_H time guard (default 11.3)
#      T4_LIMIT   smoke test: first N images only (also allows CPU)
# Heads are trained here too (GPU, train labels only) -> <T4_HEADS> (default: heads/ next to OUT). Features are kept.
# Blend + clean submission afterwards (local): python work/t4_ms_pipeline.py <heads dir>
import os, glob, time, json, subprocess, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F, timm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Priority = weight in the submitted frozen set (v37 / v59r3): ViT-H+ .20 > DINOv2-g .15 > RADIOv3-H .10 > RADIOv4-H .05.
# One Kaggle session (12 h, T4) fits about 10 h of work: see the session plan in the header. Edit this default in the Kaggle UI.
ONLY = [b for b in os.environ.get("T4_FEATS", "vithp,dinov2g,radio3,radio").split(",") if b]
SCALES = [s for s in os.environ.get("T4_SCALES", "s05,s10,s15").split(",") if s]
OUT = os.environ.get("T4_OUT", "/kaggle/working/feats"); os.makedirs(OUT, exist_ok=True)
LIMIT = int(os.environ.get("T4_LIMIT", "0")); TIME_LIMIT_H = float(os.environ.get("T4_TIME_H", "11.3"))
if not os.environ.get("HF_TOKEN"):                    # gated DINOv3 weights: Kaggle Secrets (UI run) or Colab secrets
    for get in (lambda: __import__("kaggle_secrets").UserSecretsClient().get_secret("HF_TOKEN"),
                lambda: __import__("google.colab.userdata", fromlist=["get"]).get("HF_TOKEN")):
        try:
            t = get()
            if t: os.environ["HF_TOKEN"] = t; break
        except Exception: pass
print("HF_TOKEN", "found" if os.environ.get("HF_TOKEN") else "not found (vithp will be skipped)", flush=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda" or LIMIT, "turn on the GPU (CPU is only allowed for a T4_LIMIT smoke test)"
AMP = torch.float16 if DEVICE == "cuda" else torch.bfloat16
T_START = time.time()
print(DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "", "| timm", timm.__version__, "| torch", torch.__version__, flush=True)
BASE = {"s05": (112, 208), "s10": (224, 432), "s15": (336, 640)}          # (H, W); snapped to each model's patch grid below
LAYERS = {"l50": 0.5, "l75": 0.75}
BS = {"s05": 64, "s10": 32, "s15": 16}

root = os.environ.get("T4_DATA_ROOT")
for _ in range(30):
    if root: break
    hits = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train.csv", recursive=True)]
    hits = [h for h in hits if os.path.isdir(f"{h}/train") and os.path.isdir(f"{h}/test") and os.path.isfile(f"{h}/test.csv")]
    if hits: root = sorted(hits)[0]; break
    time.sleep(10)
assert root, "task4 image data not found (set T4_DATA_ROOT)"
train = pd.read_csv(f"{root}/train.csv"); test = pd.read_csv(f"{root}/test.csv")
def stem(df):
    c = "file_name" if "file_name" in df.columns else "id"
    return df[c].astype(str).str.replace(".jpg", "", regex=False).values
tr_names, te_names = stem(train), stem(test)
if LIMIT: tr_names, te_names = tr_names[:LIMIT], te_names[:LIMIT]
np.save(f"{OUT}/train_names.npy", tr_names); np.save(f"{OUT}/test_names.npy", te_names)
print(root, len(tr_names), len(te_names), flush=True)

def load_all(folder, names):
    read = lambda i: np.asarray(Image.open(f"{folder}/{i}.jpg").convert("RGB").resize((230, 120)))
    with ThreadPoolExecutor(16) as ex: return torch.from_numpy(np.stack(list(ex.map(read, names, chunksize=64))))
t0 = time.time(); X = {"train": load_all(f"{root}/train", tr_names), "test": load_all(f"{root}/test", te_names)}
print("loaded", {k: tuple(v.shape) for k, v in X.items()}, f"{time.time()-t0:.0f}s", flush=True)


def find_blocks(m):
    """Longest ModuleList/Sequential whose name ends in 'blocks' = the transformer stack (timm ViT and RADIO's inner ViT)."""
    best = None
    for n, mod in m.named_modules():
        if n.split(".")[-1] == "blocks" and isinstance(mod, (torch.nn.ModuleList, torch.nn.Sequential)) and (best is None or len(mod) > len(best)):
            best = mod
    return best


def snap(size, p): return (max(p, round(size[0] / p) * p), max(p, round(size[1] / p) * p))


# ---- backbones: maker() -> (model, patch size, fn(x in [0,1]) -> B x D, res(size) -> supported size) -------------------
def make_timm(name, pretrained=True):
    m = timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=(224, 432), dynamic_img_size=True).to(DEVICE).eval()
    cfg = timm.data.resolve_model_data_config(m); p = m.patch_embed.patch_size[0]
    mean = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)
    def fn(x):
        t = m.forward_features((x - mean) / std)
        return torch.cat([t[:, 0], t[:, m.num_prefix_tokens:].mean(1)], 1)
    return m, p, fn, lambda s: snap(s, p)


def make_radio(repo):
    for mod, pkg in [("einops", "einops"), ("open_clip", "open_clip_torch")]:      # RADIO remote code imports both
        try: __import__(mod)
        except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
    from transformers import AutoModel
    m = AutoModel.from_pretrained(repo, trust_remote_code=True).to(DEVICE).eval()
    p = int(getattr(m, "patch_size", 16) or 16)
    def res(s):
        if hasattr(m, "get_nearest_supported_resolution"):
            r = m.get_nearest_supported_resolution(*s)
            return (int(r.height), int(r.width)) if hasattr(r, "height") else (int(r[0]), int(r[1]))
        return snap(s, p)
    def fn(x):                                         # RADIO takes [0, 1] input and normalises internally
        out = m(x); summary, spatial = (out[0], out[1]) if isinstance(out, (tuple, list)) else (out.summary, out.features)
        return torch.cat([summary.flatten(1), spatial.mean(1)], 1)
    return m, p, fn, res


MAKERS = {"radio3": lambda: make_radio("nvidia/C-RADIOv3-H"), "radio": lambda: make_radio("nvidia/C-RADIOv4-H"),
          "radio3g": lambda: make_radio("nvidia/C-RADIOv3-g"), "radio3b": lambda: make_radio("nvidia/C-RADIOv3-B"),
          "dinov2g": lambda: make_timm("vit_giant_patch14_reg4_dinov2.lvd142m"),
          "vithp": lambda: make_timm("vit_huge_plus_patch16_dinov3.lvd1689m"),
          "tiny": lambda: make_timm("vit_tiny_patch16_224", pretrained=False)}        # smoke test only (random weights)


def ftag(bk, s): return f"{bk}_{s}"
def done(bk, s, split, v): return os.path.exists(f"{OUT}/{ftag(bk, s)}_{split}_{v}.npy")
def save(bk, s, split, v, A):
    assert np.isfinite(A).all(), f"{bk} {s} {split} {v}: non-finite features"
    f = f"{OUT}/{ftag(bk, s)}_{split}_{v}.npy"; np.save(f + ".tmp.npy", A); os.replace(f + ".tmp.npy", f)


@torch.no_grad()
def run_pass(fn, blocks, P, size, Xs, v, bs, want_layers):
    """One pass over Xs at `size`. Returns the final features and, if want_layers, the hooked intermediate-block features."""
    store, hooks = {}, []
    if want_layers:
        n_patch = (size[0] // P) * (size[1] // P)
        for lt, frac in LAYERS.items():
            i = min(len(blocks) - 1, max(0, round(len(blocks) * frac) - 1))
            def hook(_m, _in, out, lt=lt):
                t = out[0] if isinstance(out, (tuple, list)) else out
                store[lt] = torch.cat([t[:, 0], t[:, -n_patch:].mean(1)], 1).float()      # [first token, mean of patch tokens]
            hooks.append(blocks[i].register_forward_hook(hook))
    fin, lay = [], ({lt: [] for lt in LAYERS} if want_layers else {})
    try:
        i = 0
        while i < len(Xs):
            try:
                x = Xs[i:i + bs].to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
                x = F.interpolate(x, size=size, mode="bicubic", align_corners=False).clamp_(0, 1)
                if v == "hflip": x = x.flip(3)
                with torch.autocast(DEVICE, dtype=AMP): f = fn(x)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); bs = max(1, bs // 2); print("  OOM -> batch", bs, flush=True); continue
            fin.append(f.float().cpu().numpy().astype(np.float16))
            for lt in lay: lay[lt].append(store[lt].cpu().numpy().astype(np.float16))
            i += x.shape[0]
    finally:
        for h in hooks: h.remove()
    return np.concatenate(fin), {lt: np.concatenate(a) for lt, a in lay.items()}


# ---- heads (train labels only): same linear head as work/t4_head_lowmem.py ------------------------------------------------
# per tag: {bk}_s05 / _s10 / _s15 / _l50 / _l75;  early fusion (features concatenated, each standardised with its own
# train-plain statistics): {bk}_ms = s05+s10+s15, {bk}_msl = s05+s10+s15+l50+l75.
# Output <HEADS>/{name}/d3lin_{name}_{oof,test_fold{k}}.npy; test rows in the order of HEADS/test_names.npy (= this run's
# test.csv order), oof rows in train order. Fold split, weight-decay grid and epochs are fixed (no test information).
HEADS = os.environ.get("T4_HEADS", os.path.join(os.path.dirname(OUT.rstrip("/")) or ".", "heads"))
CLASSES = ["7-11k", "fiveN", "freefried", "pube", "roblock"]                 # datacard order = alphabetical
if "label" in train.columns: Y = train["label"].values.astype(int)
else: Y = train["class"].map({c: i for i, c in enumerate(CLASSES)}).values.astype(int)
if LIMIT: Y = Y[:LIMIT]


def train_heads(bk):
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import f1_score
    os.makedirs(HEADS, exist_ok=True); np.save(f"{HEADS}/test_names.npy", te_names); np.save(f"{HEADS}/train_names.npy", tr_names)
    lay = list(LAYERS) if "s10" in SCALES else []
    groups = {ftag(bk, s): [s] for s in SCALES + lay}
    if len(SCALES) > 1: groups[f"{bk}_ms"] = list(SCALES)
    if lay: groups[f"{bk}_msl"] = list(SCALES) + lay
    K = len(CLASSES); y = torch.from_numpy(Y).to(DEVICE)
    nmin = np.bincount(Y, minlength=K).min()
    folds = list((StratifiedKFold(5, shuffle=True, random_state=42).split(Y, Y)) if nmin >= 5 else KFold(5, shuffle=True, random_state=42).split(Y))
    for name, parts in groups.items():
        if os.path.exists(f"{HEADS}/{name}/d3lin_{name}_oof.npy"): print("head done:", name, flush=True); continue
        t0 = time.time(); Xtr, Xte = {}, {}
        for split, store in (("train", Xtr), ("test", Xte)):
            for v in ("plain", "hflip"):
                cols = []
                for s in parts:
                    A = np.load(f"{OUT}/{ftag(bk, s)}_{split}_{v}.npy", mmap_mode="r")
                    R = np.load(f"{OUT}/{ftag(bk, s)}_train_plain.npy", mmap_mode="r")
                    mu = np.asarray(R, dtype=np.float32).mean(0); sd = np.asarray(R, dtype=np.float32).std(0) + 1e-6
                    cols.append(torch.from_numpy(((np.asarray(A, dtype=np.float32) - mu) / sd).astype(np.float16)))
                store[v] = torch.cat(cols, 1).to(DEVICE)
        d = Xtr["plain"].shape[1]
        def fit(idx, wd, epochs=40):
            idx_t = torch.from_numpy(idx).to(DEVICE); n2 = 2 * len(idx)
            cw = torch.tensor(1 / np.sqrt(np.bincount(Y[idx], minlength=K) + 1e-9), dtype=torch.float32, device=DEVICE); cw = cw / cw.mean()
            m = torch.nn.Linear(d, K).to(DEVICE); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=wd)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); lf = torch.nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
            for _ in range(epochs):
                p = torch.randperm(n2, device=DEVICE)
                for i in range(0, n2, 1024):
                    b = p[i:i + 1024]; vi = b >= len(idx); r = idx_t[b % len(idx)]
                    xb = torch.where(vi[:, None], Xtr["hflip"][r], Xtr["plain"][r]).float()
                    opt.zero_grad(); lf(m(xb), y[r]).backward(); opt.step()
                sch.step()
            return m
        @torch.no_grad()
        def pred(m, A, idx=None):
            A = A if idx is None else A[torch.from_numpy(idx).to(DEVICE)]
            return torch.cat([torch.softmax(m(A[i:i + 8192].float()), 1) for i in range(0, len(A), 8192)]).cpu().numpy()
        ev = lambda m, idx: (pred(m, Xtr["plain"], idx) + pred(m, Xtr["hflip"], idx)) / 2
        a, b = folds[0]; ia, ib = a[: int(0.8 * len(a))], a[int(0.8 * len(a)):]
        wd = max([1e-4, 1e-2, 1e-1], key=lambda w: f1_score(Y[ib], ev(fit(ia, w, 20), ib).argmax(1), average="macro"))
        od = f"{HEADS}/{name}"; os.makedirs(od, exist_ok=True); oof = np.zeros((len(Y), K), np.float32)
        for k, (a, b) in enumerate(folds):
            m = fit(a, wd); oof[b] = ev(m, b)
            np.save(f"{od}/d3lin_{name}_test_fold{k}.npy", ((pred(m, Xte["plain"]) + pred(m, Xte["hflip"])) / 2).astype(np.float32))
        np.save(f"{od}/d3lin_{name}_oof.npy", oof)
        f1 = f1_score(Y, oof.argmax(1), average="macro"); log[f"head_{name}"] = [d, wd, round(float(f1), 5), round(time.time() - t0)]
        json.dump(log, open(log_p, "w"), indent=1)
        print(f"head {name}: dim {d} wd {wd} random-CV OOF macroF1 {f1:.5f} {time.time()-t0:.0f}s", flush=True)
        del Xtr, Xte
        if DEVICE == "cuda": torch.cuda.empty_cache()


log_p = f"{OUT}/log.json"; log = json.load(open(log_p)) if os.path.exists(log_p) else {}
for bk in ONLY:
    todo = [(s, sp, v) for s in SCALES for sp in ("test", "train") for v in ("plain", "hflip")
            if not done(bk, s, sp, v) or (s == "s10" and not all(done(bk, lt, sp, v) for lt in LAYERS))]
    if not todo: print("features done:", bk, flush=True); train_heads(bk); continue
    if (time.time() - T_START) / 3600 > TIME_LIMIT_H: print("time guard: stop before", bk, flush=True); break
    try:
        m, P, fn, res = MAKERS[bk](); blocks = find_blocks(m)
        print(bk, f"{sum(q.numel() for q in m.parameters())/1e6:.0f}M params | patch {P} | blocks {None if blocks is None else len(blocks)}",
              "| sizes", {s: res(BASE[s]) for s in SCALES}, flush=True)
    except Exception as e:
        print(f"skip {bk}: {type(e).__name__}: {str(e)[:300]}", flush=True); log[bk] = f"failed: {e}"; continue
    rate = {}                                          # measured seconds per (image x patch token) of this backbone
    for s, sp, v in todo:
        size = res(BASE[s]); wl = s == "s10" and blocks is not None; ntok = (size[0] // P) * (size[1] // P) * len(X[sp])
        est = ntok * max(rate.values()) * 1.2 if rate else 0
        if (time.time() - T_START + est) / 3600 > TIME_LIMIT_H:
            print(f"time guard: stop at {bk} {s} {sp} {v} (next pass about {est/60:.0f} min)", flush=True); break
        t0 = time.time()
        A, L = run_pass(fn, blocks, P, size, X[sp], v, BS[s], wl)
        rate[s] = (time.time() - t0) / ntok
        save(bk, s, sp, v, A); log[f"{ftag(bk, s)}_{sp}_{v}"] = [list(A.shape), list(size), round(time.time() - t0)]
        for lt, B in L.items(): save(bk, lt, sp, v, B); log[f"{ftag(bk, lt)}_{sp}_{v}"] = [list(B.shape), list(size), 0]
        json.dump(log, open(log_p, "w"), indent=1)
        print(bk, s, sp, v, A.shape, {lt: B.shape for lt, B in L.items()}, size, f"{time.time()-t0:.0f}s", flush=True)
    del m, fn
    if DEVICE == "cuda": torch.cuda.empty_cache()
    if all(done(bk, s, sp, v) for s in SCALES + (list(LAYERS) if "s10" in SCALES else []) for sp in ("test", "train") for v in ("plain", "hflip")):
        train_heads(bk)                                # features stay on disk; heads are trained right away (timeout-safe)
print("done - nothing submitted", flush=True)
