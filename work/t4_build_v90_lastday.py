"""Build 5 research-backed, CPU-only, 100% train-only (TA-rule compliant) last-day Task 4 submissions (v90a..v90e).

Does NOT submit to Kaggle. Keeps T1/T2/T3/T5 identical to sub_v85a_soup_trainprior.csv (0.87346).
Outputs:
  submissions/FINAL/sub_v90a_multidepth_superbase.csv
  submissions/FINAL/sub_v90b_jointmb_lora_blend.csv
  submissions/FINAL/sub_v90c_tempcal_multidepth_sb.csv
  submissions/FINAL/sub_v90d_entropy_adaptive_md.csv
  submissions/FINAL/sub_v90e_powermean_non7b_diverse.csv
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, log_loss
from scipy.optimize import minimize_scalar

torch.set_num_threads(os.cpu_count() or 4)
T0 = time.time()

tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv")
C = sorted(tr["class"].unique())
K = len(C)
y = tr["class"].map({c: i for i, c in enumerate(C)}).values
pt = tr["class"].value_counts(normalize=True).reindex(C).values
tr_ids = tr["id"].astype(str).str.replace(".jpg", "", regex=False).values
ref = pd.read_csv("t4/preds/v3_5f/preds/task4_v3_5f.csv")
ref_imgs = ref["id_img"].astype(str).values

new_te = pd.read_csv("cpe342-kaggle-main/04_image_classification/test.csv")
new_imgs = new_te["file_name"].str.replace(".jpg", "", regex=False).values
old_to_new = pd.Series(np.arange(len(ref)), index=ref_imgs).loc[new_imgs].values

FOLD = np.load("t4/research/groupcv/sgkf_sig8_fold.npy")
va0 = np.load("t4/preds/v4f_lora7b/preds/lora7b_l75_val0_val_idx.npy")
tr0 = np.where(FOLD != 0)[0]
yv0 = y[va0]
assert (np.where(FOLD == 0)[0] == va0).all()

lg = lambda p: np.log(np.clip(p, 1e-9, 1.0))
sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
norm = lambda p: p / np.clip(p.sum(1, keepdims=True), 1e-12, None)
ent = lambda p: -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(1)

# ==============================================================================
# STEP 1: Train & Cache 10-Backbone Early-Fusion Joint Linear Probe (AllFam)
#         (Convex L2 linear head, per-backbone train-mean centering + L2 norm)
# ==============================================================================
JDIR = "t4/preds/joint_mb_linear"
os.makedirs(JDIR, exist_ok=True)
JF_VAL = f"{JDIR}/allfam_joint_wd015_val0.npy"
JF_TE = f"{JDIR}/allfam_joint_wd015_test.npy"

ALLFAM_SPECS = [
    ("7b_l75", "t4/preds/frozen_7b_ms/feats", "vit7b_l75"),
    ("7b_s15", "t4/preds/frozen_7b_ms/feats", "vit7b_s15"),
    ("7b_s10", "t4/preds/frozen_7b/feats", "vit_7b_patch16_dinov3"),
    ("7b_l50", "t4/preds/frozen_7b_ms/feats", "vit7b_l50"),
    ("radio3g", "t4/preds/frozen_e/feats", "radio3g"),
    ("radio3", "t4/preds/frozen_c/feats", "radio3"),
    ("radio", "t4/preds/frozen_b/feats", "radio"),
    ("dinov2g", "t4/preds/frozen_c/feats", "dinov2g"),
    ("vithp", "t4/preds/dinov3_feats/feats", "vithp"),
    ("webssl1b", "t4/preds/frozen_e/feats", "webssl1b"),
]

def load_allfam(mean_idx, need_test=False):
    tr_parts, te_parts = [], []
    for nm, d, tag in ALLFAM_SPECS:
        ftr = f"{d}/{tag}_train_plain.npy"
        if nm == "vithp" and not os.path.exists(ftr):
            tag = "vit_huge_plus_patch16_dinov3"
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
        tr_parts.append(X_tr.astype(np.float16))
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
            te_parts.append(X_te.astype(np.float16))
            del X_te
    return np.concatenate(tr_parts, axis=1), (np.concatenate(te_parts, axis=1) if need_test else None)

def train_joint_linear(X_tr_f16, idx_fit, X_eval_f16, wd=0.15, epochs=30, lr=1e-2, ls=0.05, seed=42):
    torch.manual_seed(seed)
    d = X_tr_f16.shape[1]
    Xa = torch.from_numpy(X_tr_f16[idx_fit])
    Ya = torch.from_numpy(y[idx_fit]).long()
    cw = torch.tensor(1.0 / np.sqrt(np.bincount(y[idx_fit], minlength=K)), dtype=torch.float32)
    cw = cw / cw.mean()
    m = torch.nn.Linear(d, K)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf = torch.nn.CrossEntropyLoss(weight=cw, label_smoothing=ls)
    for _ in range(epochs):
        perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            lf(m(Xa[b].float()), Ya[b]).backward()
            opt.step()
        sch.step()
    m.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X_eval_f16), 4096):
            outs.append(torch.softmax(m(torch.from_numpy(X_eval_f16[i:i + 4096]).float()), 1).numpy())
    return np.concatenate(outs, axis=0)

if not os.path.exists(JF_VAL):
    print("[1/2] Training AllFam_Joint_Linear (56,576-d, wd=0.15) on Scene Fold 0...", flush=True)
    X_tr0, _ = load_allfam(tr0, need_test=False)
    P_jval = train_joint_linear(X_tr0, tr0, X_tr0[va0], wd=0.15, epochs=30, seed=42)
    np.save(JF_VAL, P_jval.astype(np.float32))
    del X_tr0
    print(f"  Fold-0 AllFam_Joint_Linear: raw F1={f1_score(yv0, P_jval.argmax(1), average='macro'):.5f} | +prior F1={f1_score(yv0, (P_jval/pt).argmax(1), average='macro'):.5f} ({int(((P_jval/pt).argmax(1)!=yv0).sum())} err) [{time.time()-T0:.0f}s]", flush=True)
else:
    P_jval = np.load(JF_VAL)
    print(f"[1/2] Loaded cached {JF_VAL} | +prior F1={f1_score(yv0, (P_jval/pt).argmax(1), average='macro'):.5f}", flush=True)

if not os.path.exists(JF_TE):
    print("[2/2] Training AllFam_Joint_Linear (56,576-d, wd=0.15, 2-seed bag) on FULL train -> test...", flush=True)
    all_idx = np.arange(len(tr))
    X_all, X_te = load_allfam(all_idx, need_test=True)
    p1 = train_joint_linear(X_all, all_idx, X_te, wd=0.15, epochs=30, seed=42)
    p2 = train_joint_linear(X_all, all_idx, X_te, wd=0.15, epochs=30, seed=7)
    P_jte = 0.5 * (p1 + p2)
    np.save(JF_TE, P_jte.astype(np.float32))
    del X_all, X_te
    print(f"  Saved {JF_TE} [{time.time()-T0:.0f}s]", flush=True)
else:
    P_jte = np.load(JF_TE)
    print(f"[2/2] Loaded cached {JF_TE}", flush=True)

# ==============================================================================
# STEP 2: Compute Strict Train-Only (Fold-0) OOF Calibration Temperatures
# ==============================================================================
def load_val_lora(d, t):
    return sm(np.mean([lg(np.load(f"{d}/lora7b_{t}_oof_{v}.npy")) for v in ("plain", "hflip")], 0))

V_s2 = load_val_lora("t4/preds/v4f_lora7b_q4", "soup2_val0_refit")
V_c24 = load_val_lora("t4/preds/v4f_lora7b_q1/preds", "c24f12r16e3_val0")
V_c40 = load_val_lora("t4/preds/v4f_lora7b_q1/preds", "c40f28r16e3_val0")
V_cnxts = np.mean([np.load(f"t4/preds/scenecv_s/preds/v4g_cnxt_s_d3_base_oof_{v}.npy") for v in ("plain", "hflip", "up")], 0)[va0]
V_7bl75 = np.load("t4/research/groupcv/oof_sgkf/vit7b_l75.npy")[va0]
V_7bs15 = np.load("t4/research/groupcv/oof_sgkf/vit7b_s15.npy")[va0]
V_7bs10 = np.load("t4/research/groupcv/oof_sgkf/vit7b.npy")[va0]
V_7bs05 = np.load("t4/research/groupcv/oof_sgkf/vit7b_s05.npy")[va0]
V_vhp = np.load("t4/research/groupcv/oof_sgkf/vithp.npy")[va0]
V_r4 = np.load("t4/research/groupcv/oof_sgkf/radio.npy")[va0]
V_r3 = np.load("t4/research/groupcv/oof_sgkf/radio3.npy")[va0]
V_d2g = np.load("t4/research/groupcv/oof_sgkf/dinov2g.npy")[va0]

V_sb_proxy = sm(
    0.55 * lg(V_cnxts)
    + 0.25 * (0.55 * lg(V_7bl75) + 0.20 * lg(V_7bs15) + 0.15 * lg(V_7bs10) + 0.10 * lg(V_7bs05))
    + 0.20 * (0.25 * lg(V_vhp) + 0.25 * lg(V_r3) + 0.25 * lg(V_r4) + 0.25 * lg(V_d2g))
)

fit_T = lambda P: float(minimize_scalar(lambda T: log_loss(yv0, sm(lg(P) / T)), bounds=(0.1, 3.0), method="bounded").x)
T_s2 = fit_T(V_s2)
T_c24 = fit_T(V_c24)
T_c40 = fit_T(V_c40)
T_sb = fit_T(V_sb_proxy)
T_jmb = fit_T(P_jval)
print(f"\nStrict Fold-0 NLL-optimal temperatures: T_s2={T_s2:.3f}, T_c24={T_c24:.3f}, T_c40={T_c40:.3f}, T_sb={T_sb:.3f}, T_jmb={T_jmb:.3f}")

# Verify Fold-0 metrics of our 5 methods (with Fold-0 proxies)
rng = np.random.default_rng(2026)
dir_priors = rng.dirichlet(np.ones(K) * 1.5, size=200)
def eval_f0(name, P):
    pr = P.argmax(1)
    cm = np.bincount(yv0 * K + pr, minlength=K * K).reshape(K, K).astype(float)
    cond = cm / np.maximum(cm.sum(1, keepdims=True), 1.0)
    uni = float((2 * np.diag(cond / K) / np.maximum((cond / K).sum(0) + (cond / K).sum(1), 1e-12)).mean())
    d200 = float(np.mean([(2 * np.diag(cond * pi[:, None]) / np.maximum((cond * pi[:, None]).sum(0) + (cond * pi[:, None]).sum(1), 1e-12)).mean() for pi in dir_priors]))
    print(f"  {name:44s} | err={int((pr!=yv0).sum()):2d} | F1={f1_score(yv0, pr, average='macro'):.5f} | uni={uni:.5f} | dir200={d200:.5f}")

V_c78_proxy = sm(0.5 * lg(V_cnxts) + 0.425 * lg(V_7bl75) + 0.01875 * (lg(V_vhp) + lg(V_r4) + lg(V_r3) + lg(V_d2g)))
print("\n=== SCENE FOLD 0 VALIDATION (5,113 held-out scene images) ===")
eval_f0("Baseline v85a proxy (0.85 s2 + 0.15 c78)", norm(sm(0.85*lg(V_s2) + 0.15*lg(V_c78_proxy)) / pt))
eval_f0("v90a: MultiDepth + SuperBase", norm(sm(0.77*lg(V_s2) + 0.04*lg(V_c24) + 0.04*lg(V_c40) + 0.15*lg(V_sb_proxy)) / pt))
eval_f0("v90b: JointMB + MultiDepth + SuperFT", norm(sm(0.78*lg(V_s2) + 0.04*lg(V_c24) + 0.04*lg(V_c40) + 0.07*lg(P_jval) + 0.07*lg(V_cnxts)) / pt))
T_eff_c = 1.0 / (0.77/T_s2 + 0.04/T_c24 + 0.04/T_c40 + 0.15/T_sb)
eval_f0("v90c: OOF-TempCal MultiDepth + SuperBase", sm((0.77*lg(V_s2)/T_s2 + 0.04*lg(V_c24)/T_c24 + 0.04*lg(V_c40)/T_c40 + 0.15*lg(V_sb_proxy)/T_sb) - (1.0/T_eff_c)*np.log(pt)))
# v90d entropy-adaptive on fold 0
def ew_cal_pool(items, gam=1.0):
    num, t_num = np.zeros_like(items[0][0]), np.zeros((len(items[0][0]), 1))
    for P, T_m, w in items:
        Pcal = sm(lg(P) / T_m)
        wt = w * np.exp(-gam * ent(Pcal))[:, None]
        num += wt * (lg(P) / T_m)
        t_num += wt / T_m
    return sm(num - t_num * np.log(pt))
eval_f0("v90d: Cal-EntropyWeighted MultiDepth+SB+JMB", ew_cal_pool([(V_s2, T_s2, 0.77), (V_c24, T_c24, 0.04), (V_c40, T_c40, 0.04), (V_sb_proxy, T_sb, 0.09), (P_jval, T_jmb, 0.06)], gam=1.0))
def power_pool(items, q=0.35):
    return norm(sum(w * np.clip(norm(P / pt), 1e-9, 1.0) ** q for P, w in items) ** (1.0 / q))
V_non7b_p = sm(0.65*lg(V_cnxts) + 0.35*(0.25*lg(V_vhp)+0.25*lg(V_r3)+0.25*lg(V_r4)+0.25*lg(V_d2g)))
eval_f0("v90e: PowerMean(0.35) Non7B+JMB+MultiDepth", power_pool([(V_s2, 0.77), (V_c24, 0.04), (V_c40, 0.04), (V_non7b_p, 0.09), (P_jval, 0.06)], q=0.35))

# ==============================================================================
# STEP 3: Load ALL Test Models & Build the 5 Final Last-Day Submissions
# ==============================================================================
V3 = ["plain", "hflip", "up"]
def load_ft(d, t, vs):
    v0 = vs[0] if vs else "plain"
    if os.path.exists(f"{d}/{t}_test_full_{v0}.npy"):
        return np.mean([np.load(f"{d}/{t}_test_full_{v}.npy") for v in vs], 0)
    o = []
    for k in range(5):
        fs = [f"{d}/{t}_test_fold{k}_{v}.npy" for v in vs] if vs else [f"{d}/{t}_test_fold{k}.npy"]
        if os.path.exists(fs[0]):
            o.append(np.mean([np.load(x) for x in fs], 0))
    return np.mean(o, 0)

def load_fr(dname):
    fs = [f"t4/preds/dinov3_head/{dname}/d3lin_{dname}_test_fold{k}.npy" for k in range(5)]
    return np.mean([np.load(x) for x in fs if os.path.exists(x)], 0)

def load_lora(d, tag):
    return sm(np.mean([lg(np.load(f"{d}/{tag}_test_full_{v}.npy")) for v in ("plain", "hflip")], 0))

Ps2 = load_lora("t4/preds/v4f_lora7b_q4", "v4f_lora7b_soup2_s42")
Pc24 = load_lora("t4/preds/v4f_lora7b_q2/preds", "v4f_lora7b_c24f12r16e3_s42")
Pc40 = load_lora("t4/preds/v4f_lora7b_q1/preds", "v4f_lora7b_c40f28r16e3_s42")

ft_vitL = load_ft("t4/preds/v4f_vit_l_d3", "v4f_vit_l_d3_v4aug_s42", V3)
ft_evaL = load_ft("t4/preds/v4f_eva02_l", "v4f_eva02_l_v4aug_s42", V3)
ft_evaB = load_ft("t4/preds/v4_eva02_b/preds", "v4_eva02_b", V3)
ft_cnxtLf = load_ft("t4/preds/v4f_cnxt_l_d3c", "v4f_cnxt_l_d3c_s42", V3)
ft_cnxtLfk = load_ft("t4/preds/v4f_cnxt_l_d3c_kg/preds", "v4f_cnxt_l_d3c_s42", V3)
ft_cnxtLv3 = load_ft("t4/preds/v4f_cnxt_l_d3c_v3aug", "v4f_cnxt_l_d3c_v3aug_s42", V3)
ft_cnxtLv3k = load_ft("t4/preds/v4f_cnxt_l_d3_v3aug_kg/preds", "v4f_cnxt_l_d3c_v3aug_s42", V3)
ft_cnxtLc = load_ft("t4/preds/v4_cnxt_l_d3_colab/preds", "v4_cnxt_l_d3c", V3)
ft_cnxtB = load_ft("t4/preds/v4_cnxt_b_d3/preds", "v4_cnxt_b_d3", V3)
ft_cnxtBf = load_ft("t4/preds/v4f_cnxt_b_d3_nolq/preds", "v4f_cnxt_b_d3_s42_nolq", V3)
ft_dinov2L = load_ft("t4/preds/v4_dinov2_l/preds", "v4_dinov2_l", V3)

fr_7bl75 = load_fr("vit7b_l75_ph")
fr_7bs15 = load_fr("vit7b_s15_ph")
fr_7bs10 = load_fr("vit_7b_patch16_dinov3_ph")
fr_7bs05 = load_fr("vit7b_s05_ph")
fr_r3g = load_fr("radio3g_ph")
fr_r3 = load_fr("radio3_ph")
fr_r4 = load_fr("radio_ph")
fr_d2g = load_fr("dinov2g_ph")
fr_vhp = load_fr("vithp_ph")
fr_web = load_fr("webssl1b_ph")

L_sft = (
    0.25 * lg(ft_vitL)
    + 0.25 * (0.75 * lg(ft_evaL) + 0.25 * lg(ft_evaB))
    + 0.25 * (0.25 * lg(ft_cnxtLf) + 0.25 * lg(ft_cnxtLfk) + 0.20 * lg(ft_cnxtLv3) + 0.15 * lg(ft_cnxtLv3k) + 0.15 * lg(ft_cnxtLc))
    + 0.15 * (0.5 * lg(ft_cnxtB) + 0.5 * lg(ft_cnxtBf))
    + 0.10 * lg(ft_dinov2L)
)
L_sfr_non7b = 0.30 * lg(fr_r3g) + 0.20 * lg(fr_vhp) + 0.18 * lg(fr_r3) + 0.14 * lg(fr_r4) + 0.13 * lg(fr_d2g) + 0.05 * lg(fr_web)
L_sfr_ms7b = 0.55 * lg(fr_7bl75) + 0.20 * lg(fr_7bs15) + 0.15 * lg(fr_7bs10) + 0.10 * lg(fr_7bs05)

P_sft = sm(L_sft)
P_sb = sm(0.55 * L_sft + 0.25 * L_sfr_ms7b + 0.20 * L_sfr_non7b)
P_non7b = sm(0.65 * L_sft + 0.35 * L_sfr_non7b)

print(f"\nAllFam_Joint_Linear test agreement with soup2: {(P_jte.argmax(1) == Ps2.argmax(1)).mean():.4f} | ent={ent(P_jte).mean():.4f}")

# Build the 5 final probability matrices (OLD order -> realigned to NEW order when writing CSV!)
FINAL_5 = {
    "sub_v90a_multidepth_superbase.csv": norm(sm(0.77 * lg(Ps2) + 0.04 * lg(Pc24) + 0.04 * lg(Pc40) + 0.15 * lg(P_sb)) / pt),
    "sub_v90b_jointmb_lora_blend.csv": norm(sm(0.78 * lg(Ps2) + 0.04 * lg(Pc24) + 0.04 * lg(Pc40) + 0.07 * lg(P_jte) + 0.07 * lg(P_sft)) / pt),
    "sub_v90c_tempcal_multidepth_sb.csv": sm((0.77 * lg(Ps2) / T_s2 + 0.04 * lg(Pc24) / T_c24 + 0.04 * lg(Pc40) / T_c40 + 0.15 * lg(P_sb) / T_sb) - (1.0 / T_eff_c) * np.log(pt)),
    "sub_v90d_entropy_adaptive_md.csv": ew_cal_pool([(Ps2, T_s2, 0.77), (Pc24, T_c24, 0.04), (Pc40, T_c40, 0.04), (P_sb, T_sb, 0.09), (P_jte, T_jmb, 0.06)], gam=1.0),
    "sub_v90e_powermean_non7b_diverse.csv": power_pool([(Ps2, 0.77), (Pc24, 0.04), (Pc40, 0.04), (P_non7b, 0.09), (P_jte, 0.06)], q=0.35),
}

# Load v85a template + 26 clean LB submissions for final audit
F = "submissions/FINAL/"
tmpl = pd.read_csv(F + "sub_v85a_soup_trainprior.csv", dtype=str, keep_default_na=False)
y85 = tmpl["task4"].astype(int).values

KNOWN_LB = {
    "sub_v74b_clean.csv": 0.86812, "sub_v78a_clean.csv": 0.86793, "sub_v78b_clean.csv": 0.86813,
    "sub_v79a_clean.csv": 0.87028, "sub_v79b_clean.csv": 0.87104, "sub_v79c_clean.csv": 0.87109,
    "sub_v79d_clean.csv": 0.87074, "sub_v80a_clean.csv": 0.87087, "sub_v81b_trainprior.csv": 0.87307,
    "sub_v82a_v80a_trainprior.csv": 0.87224, "sub_v82b_v80a_trainprior.csv": 0.87300,
    "sub_v83_lora90_trainprior.csv": 0.87298, "sub_v84a_trainprior.csv": 0.87294,
    "sub_v85a_soup_trainprior.csv": 0.87346, "sub_v85b_soup_trainprior.csv": 0.87326,
    "sub_v86a_soup2v_trainprior.csv": 0.87335, "sub_v86b_l75la.csv": 0.87219,
    "sub_v87a_clean.csv": 0.87338, "sub_v87b_clean.csv": 0.87310, "sub_v87c_clean.csv": 0.87340,
    "sub_v87d_clean.csv": 0.87342, "sub_v88_soup80_trainprior.csv": 0.87341,
    "sub_v88_soup86_trainprior.csv": 0.87345, "sub_v89a_soup85_beta085.csv": 0.87329,
    "sub_v89b_cal_soup80_beta20.csv": 0.87319, "sub_v89c_soup85_beta140.csv": 0.87345,
}
hist_preds = {fn: (pd.read_csv(F + fn)["task4"].astype(int).values, lb) for fn, lb in KNOWN_LB.items()}

print("\n=== WRITING & VERIFYING 5 NEW LAST-DAY SUBMISSIONS ===")
written = {}
for fn, P_old in FINAL_5.items():
    tag = fn.replace("sub_", "").replace(".csv", "")
    bdir = f"t4/research/blend_{tag}"
    os.makedirs(bdir, exist_ok=True)
    np.save(f"{bdir}/t4_blend_test_proba.npy", P_old.astype(np.float32))
    pd.DataFrame({"id": ref["id"], "id_img": ref_imgs, "pred_num": P_old.argmax(1), "pred_str": np.array(C)[P_old.argmax(1)]}).to_csv(
        f"{bdir}/t4_blend.csv", index=False, lineterminator="\n"
    )
    pred_new = P_old.argmax(1)[old_to_new]
    out_df = tmpl.copy()
    out_df["task4"] = pred_new.astype(int).astype(str)
    out_path = F + fn
    out_df.to_csv(out_path, index=False, lineterminator="\n")

    # Strict format & integrity checks!
    chk = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    assert chk.shape == (25889, 6), f"bad shape {chk.shape}"
    assert (chk.id.values == tmpl.id.values).all(), "id mismatch"
    assert (chk == "").sum().sum() == 0, "empty values"
    assert b"\r" not in open(out_path, "rb").read(), "CRLF found"
    for col in ("task1", "task2", "task3", "task5"):
        assert (chk[col].values == tmpl[col].values).all(), f"{col} changed!"
    assert set(chk["task4"].unique()) == {"0", "1", "2", "3", "4"}

    xs = np.array([f1_score(pred_new, yp, average="macro") for yp, _ in hist_preds.values()])
    ys = np.array([lb for _, lb in hist_preds.values()])
    b, a = np.polyfit(xs, ys, 1)
    rmse = float(np.sqrt(np.mean((ys - (a + b * xs)) ** 2)))
    lb1 = float(a + b * 1.0)
    d85 = int((pred_new != y85).sum())
    dist = np.round(np.bincount(pred_new, minlength=5) / len(pred_new), 3).tolist()
    written[fn] = pred_new
    print(f"  OK {fn:40s} | diff_v85a={d85:3d} | RMSE={rmse:.5f} | LB@1={lb1:.5f} | share={dist}")

print("\n=== PAIRWISE DIFFERENCE MATRIX AMONG THE 5 NEW SUBMISSIONS ===")
names = list(written.keys())
short = [n.split("_")[1] for n in names]
mat = pd.DataFrame([[int((written[a] != written[b]).sum()) for b in names] for a in names], index=short, columns=short)
print(mat)
print(f"\nALL DONE in {time.time()-T0:.1f}s - NOTHING SUBMITTED TO KAGGLE.")
