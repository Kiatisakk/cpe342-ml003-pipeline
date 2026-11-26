"""Train & validate 13-Backbone Weighted-Block SuperJoint models (Linear + Residual Mixup-MLP) on CPU.

Adds PE-Core-L, SigLIP2 SO400M (vision-language!), and 7b_s05 to the 10 foundation models,
scales each L2-normalized backbone block by its Scene-CV strength, and trains on both plain & hflip views.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, log_loss

torch.set_num_threads(os.cpu_count() or 4)
T0 = time.time()

tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv")
C = sorted(tr["class"].unique()); K = len(C)
y = tr["class"].map({c: i for i, c in enumerate(C)}).values
pt = tr["class"].value_counts(normalize=True).reindex(C).values
tr_ids = tr["id"].astype(str).str.replace(".jpg", "", regex=False).values
ref = pd.read_csv("t4/preds/v3_5f/preds/task4_v3_5f.csv")
ref_imgs = ref["id_img"].astype(str).values

FOLD = np.load("t4/research/groupcv/sgkf_sig8_fold.npy")
va0 = np.load("t4/preds/v4f_lora7b/preds/lora7b_l75_val0_val_idx.npy")
tr0 = np.where(FOLD != 0)[0]
yv0 = y[va0]
assert (np.where(FOLD == 0)[0] == va0).all()

# (name, dir, tag, block_scale based strictly on Scene-CV strength!)
SPECS = [
    ("7b_l75",   "t4/preds/frozen_7b_ms/feats", "vit7b_l75", 2.0),
    ("7b_s15",   "t4/preds/frozen_7b_ms/feats", "vit7b_s15", 1.35),
    ("7b_s10",   "t4/preds/frozen_7b/feats",    "vit_7b_patch16_dinov3", 1.25),
    ("7b_s05",   "t4/preds/frozen_7b_ms/feats", "vit7b_s05", 1.10),
    ("7b_l50",   "t4/preds/frozen_7b_ms/feats", "vit7b_l50", 1.00),
    ("radio3g",  "t4/preds/frozen_e/feats",     "radio3g",   1.35),
    ("radio3",   "t4/preds/frozen_c/feats",     "radio3",    1.15),
    ("vithp",    "t4/preds/dinov3_feats/feats", "vithp",     1.15),
    ("dinov2g",  "t4/preds/frozen_c/feats",     "dinov2g",   1.05),
    ("radio",    "t4/preds/frozen_b/feats",     "radio",     1.00),
    ("pe",       "t4/preds/frozen_b/feats",     "pe",        0.85),
    ("siglip2",  "t4/preds/frozen_b/feats",     "siglip2",   0.75),
    ("webssl1b", "t4/preds/frozen_e/feats",     "webssl1b",  0.75),
]

def load_weighted_13(mean_idx, need_test=False):
    tr_parts, te_parts, slices = [], [], []
    offset = 0
    for nm, d, tag, sc in SPECS:
        ftr = f"{d}/{tag}_train_plain.npy"
        if nm == "vithp" and not os.path.exists(ftr):
            tag = "vit_huge_plus_patch16_dinov3"
            ftr = f"{d}/{tag}_train_plain.npy"
        if not os.path.exists(ftr):
            print(f"SKIP {nm}: missing {ftr}", flush=True)
            continue
        assert (np.load(f"{d}/train_names.npy", allow_pickle=True) == tr_ids).all()
        A_p = np.load(f"{d}/{tag}_train_plain.npy", mmap_mode="r")
        A_h = np.load(f"{d}/{tag}_train_hflip.npy", mmap_mode="r")
        dim = A_p.shape[1]
        X_tr = np.empty((len(tr), dim), dtype=np.float32)
        for i in range(0, len(tr), 4096):
            X_tr[i:i + 4096] = 0.5 * (A_p[i:i + 4096].astype(np.float32) + A_h[i:i + 4096].astype(np.float32))
        mu = X_tr[mean_idx].mean(0, keepdims=True)
        X_tr -= mu
        X_tr /= (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-6)
        X_tr *= sc
        tr_parts.append(X_tr.astype(np.float16))
        slices.append((offset, offset + dim))
        offset += dim
        del X_tr
        if need_test:
            ten = np.load(f"{d}/test_names.npy", allow_pickle=True)
            pos = pd.Series(np.arange(len(ten)), index=ten).loc[ref_imgs].values
            T_p = np.load(f"{d}/{tag}_test_plain.npy", mmap_mode="r")
            T_h = np.load(f"{d}/{tag}_test_hflip.npy", mmap_mode="r")
            X_te = np.empty((len(ref), dim), dtype=np.float32)
            for i in range(0, len(ref), 4096):
                idx = pos[i:i + 4096]
                X_te[i:i + 4096] = 0.5 * (T_p[idx].astype(np.float32) + T_h[idx].astype(np.float32))
            X_te -= mu
            X_te /= (np.linalg.norm(X_te, axis=1, keepdims=True) + 1e-6)
            X_te *= sc
            te_parts.append(X_te.astype(np.float16))
            del X_te
    return np.concatenate(tr_parts, axis=1), (np.concatenate(te_parts, axis=1) if need_test else None), slices

def train_linear(X_tr_f16, idx_fit, X_eval_f16, slices, wd=0.20, epochs=28, lr=8e-3, ls=0.05, blk_drop=0.10, mixup=0.2, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = X_tr_f16.shape[1]
    Xa = torch.from_numpy(X_tr_f16[idx_fit])
    Ya = torch.from_numpy(y[idx_fit]).long()
    cw = torch.tensor(1.0 / np.sqrt(np.bincount(y[idx_fit], minlength=K)), dtype=torch.float32)
    cw = cw / cw.mean()
    m = nn.Linear(d, K)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf = nn.CrossEntropyLoss(weight=cw, label_smoothing=ls)
    n = len(Xa)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            xb = Xa[b].float()
            yb = Ya[b]
            # Backbone-block dropout (prevents over-reliance on any single backbone!)
            if blk_drop > 0:
                for s, e in slices:
                    if np.random.rand() < blk_drop:
                        xb[:, s:e] = 0.0
                    else:
                        xb[:, s:e] *= (1.0 / (1.0 - blk_drop))
            # Manifold mixup
            if mixup > 0 and np.random.rand() < 0.5:
                lam = float(np.random.beta(mixup, mixup))
                p2 = torch.randperm(len(xb))
                xb = lam * xb + (1.0 - lam) * xb[p2]
                logits = m(xb)
                loss = lam * lf(logits, yb) + (1.0 - lam) * lf(logits, yb[p2])
            else:
                loss = lf(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    m.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X_eval_f16), 4096):
            outs.append(torch.softmax(m(torch.from_numpy(X_eval_f16[i:i + 4096]).float()), 1).numpy())
    return np.concatenate(outs, axis=0)

class ResBottleneckHead(nn.Module):
    def __init__(self, d, h=256, k=5):
        super().__init__()
        self.skip = nn.Linear(d, k)
        self.mlp = nn.Sequential(
            nn.Dropout(0.45),
            nn.Linear(d, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(h, k),
        )
        # Zero-init last MLP layer so it starts exactly at the convex linear solution!
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, x):
        return self.skip(x) + 0.5 * self.mlp(x)

def train_resmlp(X_tr_f16, idx_fit, X_eval_f16, slices, wd=0.15, epochs=24, lr=3e-3, ls=0.05, blk_drop=0.15, mixup=0.3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = X_tr_f16.shape[1]
    Xa = torch.from_numpy(X_tr_f16[idx_fit])
    Ya = torch.from_numpy(y[idx_fit]).long()
    cw = torch.tensor(1.0 / np.sqrt(np.bincount(y[idx_fit], minlength=K)), dtype=torch.float32)
    cw = cw / cw.mean()
    m = ResBottleneckHead(d, h=256, k=K)
    opt = torch.optim.AdamW([
        {"params": m.skip.parameters(), "lr": 8e-3, "weight_decay": 0.20},
        {"params": m.mlp.parameters(), "lr": lr, "weight_decay": wd},
    ])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf = nn.CrossEntropyLoss(weight=cw, label_smoothing=ls)
    n = len(Xa)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            xb = Xa[b].float()
            yb = Ya[b]
            if blk_drop > 0:
                for s, e in slices:
                    if np.random.rand() < blk_drop:
                        xb[:, s:e] = 0.0
                    else:
                        xb[:, s:e] *= (1.0 / (1.0 - blk_drop))
            if mixup > 0 and np.random.rand() < 0.6:
                lam = float(np.random.beta(mixup, mixup))
                p2 = torch.randperm(len(xb))
                xb = lam * xb + (1.0 - lam) * xb[p2]
                logits = m(xb)
                loss = lam * lf(logits, yb) + (1.0 - lam) * lf(logits, yb[p2])
            else:
                loss = lf(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    m.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X_eval_f16), 4096):
            outs.append(torch.softmax(m(torch.from_numpy(X_eval_f16[i:i + 4096]).float()), 1).numpy())
    return np.concatenate(outs, axis=0)

OUTD = "t4/preds/superjoint13"
os.makedirs(OUTD, exist_ok=True)

print("=== STEP 1: Scene Fold 0 Validation (13 Backbones Weighted + BlockDropout + Mixup) ===", flush=True)
X_tr0, _, slices = load_weighted_13(tr0, need_test=False)
print(f"Loaded 13-backbone weighted matrix: {X_tr0.shape} ({len(slices)} blocks) [{time.time()-T0:.0f}s]", flush=True)

P_lin_v = train_linear(X_tr0, tr0, X_tr0[va0], slices, wd=0.20, epochs=28, blk_drop=0.10, mixup=0.2, seed=42)
np.save(f"{OUTD}/sj13_linear_val0.npy", P_lin_v.astype(np.float32))
print(f"  SJ13_Linear Fold-0 | raw F1={f1_score(yv0, P_lin_v.argmax(1), average='macro'):.5f} ({int((P_lin_v.argmax(1)!=yv0).sum())} err) | +prior F1={f1_score(yv0, (P_lin_v/pt).argmax(1), average='macro'):.5f} ({int(((P_lin_v/pt).argmax(1)!=yv0).sum())} err) | NLL={log_loss(yv0, P_lin_v):.4f} [{time.time()-T0:.0f}s]", flush=True)

P_mlp_v = train_resmlp(X_tr0, tr0, X_tr0[va0], slices, wd=0.15, epochs=24, blk_drop=0.15, mixup=0.3, seed=42)
np.save(f"{OUTD}/sj13_resmlp_val0.npy", P_mlp_v.astype(np.float32))
print(f"  SJ13_ResMLP Fold-0 | raw F1={f1_score(yv0, P_mlp_v.argmax(1), average='macro'):.5f} ({int((P_mlp_v.argmax(1)!=yv0).sum())} err) | +prior F1={f1_score(yv0, (P_mlp_v/pt).argmax(1), average='macro'):.5f} ({int(((P_mlp_v/pt).argmax(1)!=yv0).sum())} err) | NLL={log_loss(yv0, P_mlp_v):.4f} [{time.time()-T0:.0f}s]", flush=True)

P_ens_v = 0.5 * P_lin_v + 0.5 * P_mlp_v
print(f"  SJ13_Ens    Fold-0 | raw F1={f1_score(yv0, P_ens_v.argmax(1), average='macro'):.5f} ({int((P_ens_v.argmax(1)!=yv0).sum())} err) | +prior F1={f1_score(yv0, (P_ens_v/pt).argmax(1), average='macro'):.5f} ({int(((P_ens_v/pt).argmax(1)!=yv0).sum())} err) | NLL={log_loss(yv0, P_ens_v):.4f}", flush=True)
del X_tr0

print("\n=== STEP 2: Full-Train -> Test Predictions (2-seed bag each) ===", flush=True)
all_idx = np.arange(len(tr))
X_all, X_te, slices = load_weighted_13(all_idx, need_test=True)
print(f"Loaded FULL train+test: {X_all.shape}, {X_te.shape} [{time.time()-T0:.0f}s]", flush=True)

pl1 = train_linear(X_all, all_idx, X_te, slices, wd=0.20, epochs=28, blk_drop=0.10, mixup=0.2, seed=42)
pl2 = train_linear(X_all, all_idx, X_te, slices, wd=0.20, epochs=28, blk_drop=0.10, mixup=0.2, seed=7)
np.save(f"{OUTD}/sj13_linear_test.npy", (0.5 * (pl1 + pl2)).astype(np.float32))
print(f"  Saved sj13_linear_test.npy [{time.time()-T0:.0f}s]", flush=True)

pm1 = train_resmlp(X_all, all_idx, X_te, slices, wd=0.15, epochs=24, blk_drop=0.15, mixup=0.3, seed=42)
pm2 = train_resmlp(X_all, all_idx, X_te, slices, wd=0.15, epochs=24, blk_drop=0.15, mixup=0.3, seed=7)
np.save(f"{OUTD}/sj13_resmlp_test.npy", (0.5 * (pm1 + pm2)).astype(np.float32))
print(f"  Saved sj13_resmlp_test.npy [{time.time()-T0:.0f}s]", flush=True)
print("DONE!", flush=True)
