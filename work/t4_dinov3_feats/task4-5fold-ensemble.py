# Task 4 — frozen DINOv3 large-backbone features (no fine-tuning). NO SUBMIT: writes /kaggle/working/feats only.
# Why: the fine-tuned models score OOF 0.99 but about 0.88 on test (test images darker / more saturated). Frozen features of a
# large self-supervised backbone keep the pretrained invariances, so a linear head on them should degrade less under that shift,
# and they add diversity to the v3 ConvNeXt-T + EVA02-S blend. The head is trained locally on CPU with CV on TRAIN labels only.
# Re-uses the task4-5fold-ensemble slug because that kernel has the Kaggle secret HF_TOKEN attached (DINOv3 weights are gated).
# Output per backbone B: feats/{B}_train_{view}.npy (float16, N x 2D = [CLS, mean patch]), feats/{B}_test_{view}.npy,
#   feats/test_names.npy (test image names in the order used), feats/train_names.npy.
# Train views: plain, hflip, dark (x0.7 brightness), sat (saturation x1.6) -> photometric augmentation for the linear head.
# Test views: plain, hflip.
import os, glob, time, json
import numpy as np, pandas as pd, torch, torch.nn.functional as F, timm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN"); print("HF_TOKEN loaded")
except Exception as e:
    raise SystemExit(f"STOP: attach Kaggle secret HF_TOKEN (Add-ons -> Secrets) first ({type(e).__name__})")

BACKBONES = ["vit_large_patch16_dinov3.lvd1689m", "vit_huge_plus_patch16_dinov3.lvd1689m"]
IMG_H, IMG_W = 224, 432                                     # native 120x230 aspect, multiple of patch 16
OUT = "/kaggle/working/feats"; os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda"; assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), "| timm", timm.__version__, flush=True)

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

def view(x, v):
    if v == "hflip": return x.flip(3)
    if v == "dark": return (x * 0.7).clamp(0, 1)
    if v == "sat":
        g = x.mean(1, keepdim=True); return ((x - g) * 1.6 + g).clamp(0, 1)
    return x

@torch.no_grad()
def extract(model, X, v, mean, std, bs=64):
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i+bs].to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        x = F.interpolate(x, size=(IMG_H, IMG_W), mode="bicubic", align_corners=False).clamp_(0, 1)
        x = (view(x, v) - mean) / std
        with torch.autocast("cuda", dtype=torch.float16):
            t = model.forward_features(x)
        cls = t[:, 0]; avg = t[:, model.num_prefix_tokens:].mean(1)
        out.append(torch.cat([cls, avg], 1).float().cpu().numpy().astype(np.float16))
    return np.concatenate(out)

log = {}
for name in BACKBONES:
    tag = name.split(".")[0]
    try:
        model = timm.create_model(name, pretrained=True, num_classes=0, img_size=(IMG_H, IMG_W), dynamic_img_size=True)
    except Exception as e:
        print(f"skip {name}: {type(e).__name__}: {str(e)[:200]}", flush=True); continue
    model = model.to(DEVICE).eval()
    cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"], device=DEVICE).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=DEVICE).view(1, 3, 1, 1)
    print(name, f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params, prefix tokens {model.num_prefix_tokens}", flush=True)
    for split, X, views in [("test", TE, ["plain", "hflip"]), ("train", TR, ["plain", "hflip", "dark", "sat"])]:
        for v in views:
            t0 = time.time(); f = extract(model, X, v, mean, std)
            np.save(f"{OUT}/{tag}_{split}_{v}.npy", f)
            log[f"{tag}_{split}_{v}"] = [list(f.shape), round(time.time() - t0)]
            print(tag, split, v, f.shape, f"{time.time()-t0:.0f}s", flush=True)
            json.dump(log, open(f"{OUT}/log.json", "w"), indent=1)
    del model; torch.cuda.empty_cache()
print("done - nothing submitted")
