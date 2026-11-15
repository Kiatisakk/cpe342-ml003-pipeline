# Task 4 — Kernel B: frozen features from non-DINOv3 foundation models (no fine-tuning). NO SUBMIT: writes /kaggle/working/feats.
# Why: the frozen DINOv3 ViT-H+ head (OOF 0.964 alone) took weight 0.4 in the nested blend (0.99222 -> 0.99281) because it errs
# differently from the fine-tuned models. Backbones trained with other objectives should add more of that diversity:
#   radio   nvidia/C-RADIOv4-H (653M, distilled from DINOv3-7B + SigLIP2-g + SAM3; HF remote code, not gated)
#   siglip2 vit_so400m_patch16_siglip_384.v2_webli (400M, contrastive image-text; CLIP-family robustness to colour/light)
#   pe      vit_pe_core_large_patch14_336.fb (Meta Perception Encoder, 300M, contrastive)
# None needs HF_TOKEN, so this kernel can be pushed with the CLI.
# Output per backbone B: feats/{B}_train_{view}.npy (float16, N x [pooled, mean patch]), feats/{B}_test_{view}.npy,
#   feats/train_names.npy, feats/test_names.npy. Same views as the DINOv3 run: train plain/hflip/dark/sat, test plain/hflip.
# Head: python work/t4_dinov3_head.py <B> (train labels only, 5 folds) -> blend with work/t4_blend_runs_v1train.py.
import os, glob, time, json, subprocess, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F, timm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

ONLY = [b for b in os.environ.get("T4_FEATS", "siglip2,pe,radio").split(",") if b]
OUT = "/kaggle/working/feats"; os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda"; assert torch.cuda.is_available(), "turn on GPU"
T_START = time.time(); TIME_LIMIT_H = 10.5
print(torch.cuda.get_device_name(0), "| timm", timm.__version__, "| torch", torch.__version__, flush=True)

root = None
for _ in range(30):
    hits = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train.csv", recursive=True)]
    hits = [h for h in hits if os.path.isdir(f"{h}/train") and os.path.isdir(f"{h}/test") and os.path.isfile(f"{h}/test.csv")]
    if hits: root = sorted(hits)[0]; break
    time.sleep(10)
assert root, "task4 image data not found"
train = pd.read_csv(f"{root}/train.csv"); test = pd.read_csv(f"{root}/test.csv")
tcol = "file_name" if "file_name" in test.columns else "id"
tr_names = train["id"].astype(str).str.replace(".jpg", "", regex=False).values
te_names = test[tcol].astype(str).str.replace(".jpg", "", regex=False).values
np.save(f"{OUT}/train_names.npy", tr_names); np.save(f"{OUT}/test_names.npy", te_names)
print(root, train.shape, test.shape, "test name col", tcol, flush=True)

def load_all(folder, names):
    read = lambda i: np.asarray(Image.open(f"{folder}/{i}.jpg").convert("RGB").resize((230, 120)))
    with ThreadPoolExecutor(32) as ex: return torch.from_numpy(np.stack(list(ex.map(read, names, chunksize=64))))
t0 = time.time(); TR = load_all(f"{root}/train", tr_names); TE = load_all(f"{root}/test", te_names)
print("loaded", TR.shape, TE.shape, f"{time.time()-t0:.0f}s", flush=True)

def view(x, v):                                                # x in [0, 1]
    if v == "hflip": return x.flip(3)
    if v == "dark": return (x * 0.7).clamp(0, 1)
    if v == "sat":
        g = x.mean(1, keepdim=True); return ((x - g) * 1.6 + g).clamp(0, 1)
    return x

# ---- each backbone: (model, size, fn(x01) -> features tensor B x D) ------------------------------------------------
def make_timm(name, size):
    m = timm.create_model(name, pretrained=True, num_classes=0, img_size=size, dynamic_img_size=True).to(DEVICE).eval()
    cfg = timm.data.resolve_model_data_config(m)
    mean = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)
    def fn(x):
        t = m.forward_features((x - mean) / std)
        pooled = m.forward_head(t, pre_logits=True); avg = t[:, m.num_prefix_tokens:].mean(1)
        return torch.cat([pooled, avg], 1)
    print(name, f"{sum(p.numel() for p in m.parameters())/1e6:.0f}M", cfg["mean"], cfg["std"], flush=True)
    return m, size, fn

def make_radio(size):
    for mod, pkg in [("einops", "einops"), ("open_clip", "open_clip_torch")]:   # RADIO remote code imports both
        try: __import__(mod)
        except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
    from transformers import AutoModel
    m = AutoModel.from_pretrained("nvidia/C-RADIOv4-H", trust_remote_code=True).to(DEVICE).eval()
    if hasattr(m, "get_nearest_supported_resolution"):
        r = m.get_nearest_supported_resolution(*size); size = (int(r[0]), int(r[1])) if not hasattr(r, "height") else (r.height, r.width)
    def fn(x):                                                 # RADIO takes [0, 1] input and normalises internally
        out = m(x); summary, spatial = (out[0], out[1]) if isinstance(out, (tuple, list)) else (out.summary, out.features)
        return torch.cat([summary.flatten(1), spatial.mean(1)], 1)
    print("C-RADIOv4-H", f"{sum(p.numel() for p in m.parameters())/1e6:.0f}M, input", size, flush=True)
    return m, size, fn

MAKERS = {
    "radio":   lambda: make_radio((224, 432)),
    "siglip2": lambda: make_timm("vit_so400m_patch16_siglip_384.v2_webli", (224, 432)),
    "pe":      lambda: make_timm("vit_pe_core_large_patch14_336.fb", (224, 420)),
}

@torch.no_grad()
def extract(fn, X, v, size, bs=48):
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i+bs].to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        x = view(F.interpolate(x, size=size, mode="bicubic", align_corners=False).clamp_(0, 1), v)
        with torch.autocast("cuda", dtype=torch.float16): f = fn(x)
        out.append(f.float().cpu().numpy().astype(np.float16))
    return np.concatenate(out)

log = {}
for tag in ONLY:
    if (time.time() - T_START) / 3600 > TIME_LIMIT_H: print("time guard: stop before", tag); break
    try: m, size, fn = MAKERS[tag]()
    except Exception as e:
        print(f"skip {tag}: {type(e).__name__}: {str(e)[:300]}", flush=True); log[tag] = f"failed: {e}"; continue
    for split, X, views in [("test", TE, ["plain", "hflip"]), ("train", TR, ["plain", "hflip", "dark", "sat"])]:
        for v in views:
            t0 = time.time(); f = extract(fn, X, v, size)
            assert np.isfinite(f).all(), f"{tag} {split} {v}: non-finite features"
            np.save(f"{OUT}/{tag}_{split}_{v}.npy", f); log[f"{tag}_{split}_{v}"] = [list(f.shape), round(time.time() - t0)]
            print(tag, split, v, f.shape, f"{time.time()-t0:.0f}s", flush=True)
            json.dump(log, open(f"{OUT}/log.json", "w"), indent=1)
    del m, fn; torch.cuda.empty_cache()
print("done - nothing submitted", log)
