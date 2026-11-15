# Task 4 - Kernel E: frozen features, third batch (no fine-tuning). NO SUBMIT: writes /kaggle/working/feats.
# Why: scene-group CV (work/t4_groupcv.py) shows frozen heads from different SSL/distilled backbones make different errors on
# unseen scenes and add up (ViT-H+ + RADIOv3 + DINOv2-g 0.906 vs best single 0.893). This batch adds three more, none gated
# (no HF_TOKEN -> any account: GPU T4 + Internet on, Save & Run All):
#   radio3g   nvidia/C-RADIOv3-g (bigger sibling of radio3, the best head on 150-cluster group CV)
#   webssl1b  facebook/webssl-dino1b-full2b-224 (DINOv2 recipe trained on 2B web images: closer to screenshots than LVD)
#   aimv2h    aimv2_huge_patch14_224.apple_pt (autoregressive image+text pretraining, a different family)
# Same views and output format as Kernels B/C. Head: T4_FEATS_DIR=<feats> python work/t4_dinov3_head.py <tag>.
import os, glob, time, json, subprocess, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F, timm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

ONLY = [b for b in os.environ.get("T4_FEATS", "radio3g,webssl1b,aimv2h").split(",") if b]
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

def make_radio(repo, size):
    for mod, pkg in [("einops", "einops"), ("open_clip", "open_clip_torch")]:   # RADIO remote code imports both
        try: __import__(mod)
        except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
    from transformers import AutoModel
    m = AutoModel.from_pretrained(repo, trust_remote_code=True).to(DEVICE).eval()
    if hasattr(m, "get_nearest_supported_resolution"):
        r = m.get_nearest_supported_resolution(*size); size = (int(r[0]), int(r[1])) if not hasattr(r, "height") else (r.height, r.width)
    def fn(x):                                                 # RADIO takes [0, 1] input and normalises internally
        out = m(x); summary, spatial = (out[0], out[1]) if isinstance(out, (tuple, list)) else (out.summary, out.features)
        return torch.cat([summary.flatten(1), spatial.mean(1)], 1)
    print(repo, f"{sum(p.numel() for p in m.parameters())/1e6:.0f}M, input", size, flush=True)
    return m, size, fn

def make_hf_dinov2(repo, size):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(repo).to(DEVICE).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    def fn(x):
        h = m(pixel_values=(x - mean) / std, interpolate_pos_encoding=True).last_hidden_state      # after the final layernorm
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
    print(repo, f"{sum(p.numel() for p in m.parameters())/1e6:.0f}M, input", size, flush=True)
    return m, size, fn

def make_timm_avg(name, size):                                 # backbones without a class token: [mean, max] over patches
    m = timm.create_model(name, pretrained=True, num_classes=0, img_size=size).to(DEVICE).eval()
    cfg = timm.data.resolve_model_data_config(m)
    mean = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)
    def fn(x):
        t = m.forward_features((x - mean) / std)[:, m.num_prefix_tokens:]
        return torch.cat([t.mean(1), t.amax(1)], 1)
    print(name, f"{sum(p.numel() for p in m.parameters())/1e6:.0f}M", cfg["mean"], cfg["std"], flush=True)
    return m, size, fn

MAKERS = {
    "radio3g":  lambda: make_radio("nvidia/C-RADIOv3-g", (224, 432)),
    "webssl1b": lambda: make_hf_dinov2("facebook/webssl-dino1b-full2b-224", (224, 420)),
    "aimv2h":   lambda: make_timm_avg("aimv2_huge_patch14_224.apple_pt", (224, 420)),
}

@torch.no_grad()
def extract(fn, X, v, size, bs=32):
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
    try:
        m, size, fn = MAKERS[tag]()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):                     # smoke test before the long loop
            print(tag, "feature dim", fn(torch.rand(2, 3, *size, device=DEVICE)).shape, flush=True)
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
