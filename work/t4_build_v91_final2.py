"""Build 2 final, instant, 100% TA-rule compliant (beta_eff=1.00) last-day Task 4 submissions:
  1. submissions/FINAL/sub_v91a_entropy_dualhead_64_36.csv
       Calibrated Shannon Entropy-Weighted Dual-Head LoRA soup (0.64) + Multi-Depth (c24+c40, 0.12)
       + AllFam_Joint_Linear (0.12) + Architecture-Balanced SuperBase (0.12).
  2. submissions/FINAL/sub_v91b_hybrid_power_60_40.csv
       Zero-Avoiding Hybrid Power-Mean (q=0.35) + Calibrated Log-Pool Meta-Ensemble
       combining Dual-Head LoRA soup (0.60) + Multi-Depth (c24+c40, 0.12)
       + AllFam_Joint_Linear (0.14) + SuperBase/Non7B-SuperBase (0.14).

Does NOT submit to Kaggle. Strictly verifies format & identical T1/T2/T3/T5.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from scipy.optimize import minimize_scalar

tr = pd.read_csv("data_v1_backup/04_image_classification/train.csv")
C = sorted(tr["class"].unique()); K = len(C)
y = tr["class"].map({c: i for i, c in enumerate(C)}).values
pt = tr["class"].value_counts(normalize=True).reindex(C).values
va0 = np.load("t4/preds/v4f_lora7b/preds/lora7b_l75_val0_val_idx.npy"); yv = y[va0]
ref = pd.read_csv("t4/preds/v3_5f/preds/task4_v3_5f.csv")
ref_imgs = ref["id_img"].astype(str).values
new_te = pd.read_csv("cpe342-kaggle-main/04_image_classification/test.csv")
old_to_new = pd.Series(np.arange(len(ref)), index=ref_imgs).loc[new_te["file_name"].str.replace(".jpg", "", regex=False).values].values

lg = lambda p: np.log(np.clip(p, 1e-9, 1.0))
sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
norm = lambda p: p / np.clip(p.sum(1, keepdims=True), 1e-12, None)
ent = lambda p: -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(1)

# --- Exact Fold-0 OOF temperatures (strictly train-only!) ---
lv = lambda d, t: sm(np.mean([lg(np.load(f"{d}/lora7b_{t}_oof_{v}.npy")) for v in ("plain", "hflip")], 0))
fit_T = lambda P: float(minimize_scalar(lambda T: log_loss(yv, sm(lg(P)/T)), bounds=(0.1, 3.0), method="bounded").x)

V_s2 = lv("t4/preds/v4f_lora7b_q4", "soup2_val0_refit")
V_s2v = lv("t4/preds/v4f_lora7b_q5", "soup2b_val0_refit2v")
V_c24 = lv("t4/preds/v4f_lora7b_q1/preds", "c24f12r16e3_val0")
V_c40 = lv("t4/preds/v4f_lora7b_q1/preds", "c40f28r16e3_val0")
V_jmb = np.load("t4/preds/joint_mb_linear/allfam_joint_wd015_val0.npy")
V_cnxts = np.mean([np.load(f"t4/preds/scenecv_s/preds/v4g_cnxt_s_d3_base_oof_{v}.npy") for v in ("plain", "hflip", "up")], 0)[va0]
V_sb = sm(0.55*lg(V_cnxts)
          + 0.25*(0.55*lg(np.load("t4/research/groupcv/oof_sgkf/vit7b_l75.npy")[va0])+0.20*lg(np.load("t4/research/groupcv/oof_sgkf/vit7b_s15.npy")[va0])+0.15*lg(np.load("t4/research/groupcv/oof_sgkf/vit7b.npy")[va0])+0.10*lg(np.load("t4/research/groupcv/oof_sgkf/vit7b_s05.npy")[va0]))
          + 0.20*(0.25*lg(np.load("t4/research/groupcv/oof_sgkf/vithp.npy")[va0])+0.25*lg(np.load("t4/research/groupcv/oof_sgkf/radio3.npy")[va0])+0.25*lg(np.load("t4/research/groupcv/oof_sgkf/radio.npy")[va0])+0.25*lg(np.load("t4/research/groupcv/oof_sgkf/dinov2g.npy")[va0])))
V_n7 = sm(0.65*lg(V_cnxts) + 0.35*(0.25*lg(np.load("t4/research/groupcv/oof_sgkf/vithp.npy")[va0])+0.25*lg(np.load("t4/research/groupcv/oof_sgkf/radio3.npy")[va0])+0.25*lg(np.load("t4/research/groupcv/oof_sgkf/radio.npy")[va0])+0.25*lg(np.load("t4/research/groupcv/oof_sgkf/dinov2g.npy")[va0])))

T_s2, T_s2v, T_c24, T_c40, T_sb, T_n7, T_jmb = map(fit_T, [V_s2, V_s2v, V_c24, V_c40, V_sb, V_n7, V_jmb])

def ew_cal_pool(items, gam=1.0):
    num, t_num = np.zeros_like(items[0][0]), np.zeros((len(items[0][0]), 1))
    for P, Tm, w in items:
        wt = w * np.exp(-gam * ent(sm(lg(P)/Tm)))[:, None]
        num += wt * (lg(P)/Tm); t_num += wt / Tm
    return sm(num - t_num * np.log(pt))

def power_pool(items, q=0.35):
    return norm(sum(w * np.clip(norm(P / pt), 1e-9, 1.0)**q for P, w in items)**(1.0/q))

# Check Fold 0 first!
f0_a = ew_cal_pool([(V_s2, T_s2, 0.43), (V_s2v, T_s2v, 0.21), (V_c24, T_c24, 0.06), (V_c40, T_c40, 0.06), (V_jmb, T_jmb, 0.12), (V_sb, T_sb, 0.12)], gam=1.0)
f0_b_pw = power_pool([(V_s2, 0.40), (V_s2v, 0.20), (V_c24, 0.06), (V_c40, 0.06), (V_jmb, 0.14), (V_n7, 0.14)], q=0.35)
f0_b_ew = ew_cal_pool([(V_s2, T_s2, 0.40), (V_s2v, T_s2v, 0.20), (V_c24, T_c24, 0.06), (V_c40, T_c40, 0.06), (V_jmb, T_jmb, 0.14), (V_n7, T_n7, 0.14)], gam=1.0)
f0_b = 0.5 * f0_b_pw + 0.5 * f0_b_ew

rng = np.random.default_rng(2026)
dir_priors = rng.dirichlet(np.ones(K)*1.5, size=200)
print("=== SCENE FOLD 0 VALIDATION (5,113 unseen scenes) ===")
for nm, P in [("v91a: Cal-EntropyWeighted 64/36 (MD+JMB+SB)", f0_a), ("v91b: Hybrid PowerMean+EW 60/40 (MD+JMB+Non7B)", f0_b)]:
    pr = P.argmax(1)
    cm = np.bincount(yv*K + pr, minlength=K*K).reshape(K, K).astype(float)
    cond = cm / np.maximum(cm.sum(1, keepdims=True), 1.0)
    uni = float((2*np.diag(cond/K)/np.maximum((cond/K).sum(0)+(cond/K).sum(1), 1e-12)).mean())
    d200 = float(np.mean([(2*np.diag(cond*pi[:,None])/np.maximum((cond*pi[:,None]).sum(0)+(cond*pi[:,None]).sum(1), 1e-12)).mean() for pi in dir_priors]))
    print(f"  {nm:48s} | err={int((pr!=yv).sum()):2d} | F1={f1_score(yv, pr, average='macro'):.5f} | uni={uni:.5f} | dir200={d200:.5f}")

# --- Load TEST models ---
load_lora = lambda d, t: sm(np.mean([lg(np.load(f"{d}/{t}_test_full_{v}.npy")) for v in ("plain", "hflip")], 0))
Ps2 = load_lora("t4/preds/v4f_lora7b_q4", "v4f_lora7b_soup2_s42")
Ps2v = load_lora("t4/preds/v4f_lora7b_q5", "v4f_lora7b_soup2hv2_s42")
Pc24 = load_lora("t4/preds/v4f_lora7b_q2/preds", "v4f_lora7b_c24f12r16e3_s42")
Pc40 = load_lora("t4/preds/v4f_lora7b_q1/preds", "v4f_lora7b_c40f28r16e3_s42")
Pjmb = np.load("t4/preds/joint_mb_linear/allfam_joint_wd015_test.npy")

V3 = ["plain", "hflip", "up"]
def load_ft(d, t, vs):
    v0 = vs[0] if vs else "plain"
    if os.path.exists(f"{d}/{t}_test_full_{v0}.npy"): return np.mean([np.load(f"{d}/{t}_test_full_{v}.npy") for v in vs], 0)
    return np.mean([np.mean([np.load(f"{d}/{t}_test_fold{k}_{v}.npy") for v in vs], 0) for k in range(5) if os.path.exists(f"{d}/{t}_test_fold{k}_{v0}.npy")], 0)
load_fr = lambda dn: np.mean([np.load(f"t4/preds/dinov3_head/{dn}/d3lin_{dn}_test_fold{k}.npy") for k in range(5)], 0)

L_sft = (0.25*lg(load_ft("t4/preds/v4f_vit_l_d3","v4f_vit_l_d3_v4aug_s42",V3))
         + 0.25*(0.75*lg(load_ft("t4/preds/v4f_eva02_l","v4f_eva02_l_v4aug_s42",V3)) + 0.25*lg(load_ft("t4/preds/v4_eva02_b/preds","v4_eva02_b",V3)))
         + 0.25*(0.25*lg(load_ft("t4/preds/v4f_cnxt_l_d3c","v4f_cnxt_l_d3c_s42",V3)) + 0.25*lg(load_ft("t4/preds/v4f_cnxt_l_d3c_kg/preds","v4f_cnxt_l_d3c_s42",V3))
                 + 0.20*lg(load_ft("t4/preds/v4f_cnxt_l_d3c_v3aug","v4f_cnxt_l_d3c_v3aug_s42",V3)) + 0.15*lg(load_ft("t4/preds/v4f_cnxt_l_d3_v3aug_kg/preds","v4f_cnxt_l_d3c_v3aug_s42",V3))
                 + 0.15*lg(load_ft("t4/preds/v4_cnxt_l_d3_colab/preds","v4_cnxt_l_d3c",V3)))
         + 0.15*(0.5*lg(load_ft("t4/preds/v4_cnxt_b_d3/preds","v4_cnxt_b_d3",V3)) + 0.5*lg(load_ft("t4/preds/v4f_cnxt_b_d3_nolq/preds","v4f_cnxt_b_d3_s42_nolq",V3)))
         + 0.10*lg(load_ft("t4/preds/v4_dinov2_l/preds","v4_dinov2_l",V3)))
L_non7b = 0.30*lg(load_fr("radio3g_ph"))+0.20*lg(load_fr("vithp_ph"))+0.18*lg(load_fr("radio3_ph"))+0.14*lg(load_fr("radio_ph"))+0.13*lg(load_fr("dinov2g_ph"))+0.05*lg(load_fr("webssl1b_ph"))
L_ms7b = 0.55*lg(load_fr("vit7b_l75_ph"))+0.20*lg(load_fr("vit7b_s15_ph"))+0.15*lg(load_fr("vit_7b_patch16_dinov3_ph"))+0.10*lg(load_fr("vit7b_s05_ph"))
Psb = sm(0.55*L_sft + 0.25*L_ms7b + 0.20*L_non7b)
Pn7 = sm(0.65*L_sft + 0.35*L_non7b)

P_v91a = ew_cal_pool([(Ps2, T_s2, 0.43), (Ps2v, T_s2v, 0.21), (Pc24, T_c24, 0.06), (Pc40, T_c40, 0.06), (Pjmb, T_jmb, 0.12), (Psb, T_sb, 0.12)], gam=1.0)
P_v91b_pw = power_pool([(Ps2, 0.40), (Ps2v, 0.20), (Pc24, 0.06), (Pc40, 0.06), (Pjmb, 0.14), (Pn7, 0.14)], q=0.35)
P_v91b_ew = ew_cal_pool([(Ps2, T_s2, 0.40), (Ps2v, T_s2v, 0.20), (Pc24, T_c24, 0.06), (Pc40, T_c40, 0.06), (Pjmb, T_jmb, 0.14), (Pn7, T_n7, 0.14)], gam=1.0)
P_v91b = 0.5 * P_v91b_pw + 0.5 * P_v91b_ew

# 29-LB Diagnostic
F = "submissions/FINAL/"
tmpl = pd.read_csv(F + "sub_v85a_soup_trainprior.csv", dtype=str, keep_default_na=False)
y85 = tmpl["task4"].astype(int).values
y90f = pd.read_csv(F + "sub_v90f_entropy_dualhead_md.csv")["task4"].astype(int).values

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
    "sub_v90f_entropy_dualhead_md.csv": 0.87351, "sub_v90d_entropy_adaptive_md.csv": 0.87351,
    "sub_v90b_jointmb_lora_blend.csv": 0.87351,
}
hist = {fn: (pd.read_csv(F + fn)["task4"].astype(int).values, lb) for fn, lb in KNOWN_LB.items()}

print("\n=== WRITING & VERIFYING 2 FINAL LAST-DAY SUBMISSIONS ===")
preds_out = {}
for fn, P_old in [
    ("sub_v91a_entropy_dualhead_64_36.csv", P_v91a),
    ("sub_v91b_hybrid_power_60_40.csv", P_v91b),
]:
    tag = fn.replace("sub_", "").replace(".csv", "")
    bdir = f"t4/research/blend_{tag}"
    os.makedirs(bdir, exist_ok=True)
    np.save(f"{bdir}/t4_blend_test_proba.npy", P_old.astype(np.float32))
    pd.DataFrame({"id": ref["id"], "id_img": ref_imgs, "pred_num": P_old.argmax(1), "pred_str": np.array(C)[P_old.argmax(1)]}).to_csv(
        f"{bdir}/t4_blend.csv", index=False, lineterminator="\n"
    )
    pred_new = P_old.argmax(1)[old_to_new]
    out_df = tmpl.copy(); out_df["task4"] = pred_new.astype(int).astype(str)
    out_path = F + fn
    out_df.to_csv(out_path, index=False, lineterminator="\n")

    chk = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    assert chk.shape == (25889, 6) and (chk.id.values == tmpl.id.values).all() and (chk == "").sum().sum() == 0 and b"\r" not in open(out_path, "rb").read()
    for col in ("task1", "task2", "task3", "task5"):
        assert (chk[col].values == tmpl[col].values).all()

    xs = np.array([f1_score(pred_new, yp, average="macro") for yp, _ in hist.values()])
    ys = np.array([lb for _, lb in hist.values()])
    b, a = np.polyfit(xs, ys, 1)
    rmse = float(np.sqrt(np.mean((ys - (a + b * xs)) ** 2)))
    r_top = float(np.corrcoef(xs[ys >= 0.8732], ys[ys >= 0.8732])[0, 1])
    preds_out[fn] = pred_new
    print(f"  OK {fn:38s} | RMSE29={rmse:.5f} | LB@1={a+b:.5f} | r_top={r_top:+.3f} | diff_v90f={int((pred_new!=y90f).sum()):3d} | diff_v85a={int((pred_new!=y85).sum()):3d} | share={np.round(np.bincount(pred_new,minlength=5)/len(pred_new),3).tolist()}")

print(f"Pairwise diff v91a vs v91b: {int((preds_out['sub_v91a_entropy_dualhead_64_36.csv'] != preds_out['sub_v91b_hybrid_power_60_40.csv']).sum())} rows")
print("DONE - READY TO SUBMIT (NOT SUBMITTED).")
