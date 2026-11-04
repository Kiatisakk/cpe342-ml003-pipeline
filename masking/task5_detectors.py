"""
Task 5 — คัดฟีเจอร์ + detector หลายตระกูล (IF / kNN / LOF) + ensemble กับ TMLP
แนวทางตาม ADBench: เลือก detector ให้ตรงชนิด anomaly (global → kNN/IF, local/dependency → LOF/kNN, dependency → forecaster)

ใช้: python task5_detectors.py          (ต้องรัน task5_tmlp.py ครบก่อน เพื่อใช้ preds/task5_tmlp_scores.csv)
ไม่ submit อะไร — เขียน preds/ และ probes/ เท่านั้น
"""
import os, glob, warnings, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, K_KNN, K_LOF = 42, 10, 35
TAIL_Z, TAIL_MIN = 4.0, 0.002      # เก็บฟีเจอร์การกระโดดที่ P(|z|>4) > 0.2% (ปกติแบบ Gaussian ≈ 0.006%)
FLAGS = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
rank = lambda s: rankdata(s) / len(s)


def find(name, must=None):
    hits = glob.glob(os.path.join(HERE, "..", "**", name), recursive=True)
    if must: hits = [h for h in hits if any(m in h for m in must)]
    return sorted(hits)[0]


# ---------- 1) features ----------
raw = pd.read_csv(find("test.csv", ["05_anomaly"])); ids = pd.read_csv(find("sample_submission.csv"))["id"]
assert len(raw) == len(ids)
X = raw.drop(columns=["player_id"]); P = [1, 2, 3, 4]
bases = sorted({c.rsplit("_", 1)[0] for c in X.columns if c[-2:] in ("_1", "_2", "_3", "_4")})
NONGEO = [b for b in bases if b not in ("login_lat", "login_lon")]

F = {}
for b in NONGEO:
    v = np.stack([X[f"{b}_{p}"].values for p in P], 1)
    for k in (3, 4):
        j = v[:, k-1] - np.nanmean(v[:, :k-1], 1); m = np.nanmedian(j); s = 1.4826 * np.nanmedian(np.abs(j - m)) + 1e-9
        F[f"{b}_jz{k}"] = np.abs((j - m) / s)
w = lambda x: (x + 180) % 360 - 180
lat = np.stack([X[f"login_lat_{p}"].values for p in P], 1); lon = np.stack([X[f"login_lon_{p}"].values for p in P], 1)
home = np.full_like(lat, np.nan)
for i in range(1, 4):
    h_lat = np.nanmean(lat[:, :i], 1)
    h_lon = np.degrees(np.arctan2(np.nanmean(np.sin(np.radians(lon[:, :i])), 1), np.nanmean(np.cos(np.radians(lon[:, :i])), 1)))
    home[:, i] = np.hypot(lat[:, i] - h_lat, w(lon[:, i] - h_lon))
Fall = pd.DataFrame(F)
jz_all = list(F)
tail = (Fall > TAIL_Z).mean()
jz_keep = [c for c in jz_all if tail[c] > TAIL_MIN]

sel = Fall[jz_keep].copy()
for k in (3, 4): sel[f"geo_disp{k}"] = np.log1p(home[:, k-1])
sel["invalid_lat"] = np.nan_to_num((np.abs(lat) > 90).astype(float) * ~np.isnan(lat)).max(1)
dc = np.stack([X[f"device_count_{p}"].values for p in P], 1)
sel["device_jump4"] = dc[:, 3] - np.nanmean(dc[:, :3], 1)        # discrete → robust z = 0 จึงใช้ส่วนต่างตรง ๆ
for f in FLAGS: sel[f] = X[f].values
sel = sel.fillna(0)
for c in jz_keep: sel[c] = np.log1p(sel[c].clip(upper=50))       # บีบหางให้ระยะทาง (kNN/LOF) ไม่ถูกฟีเจอร์เดียวครอบ
for c in ("geo_disp3", "geo_disp4"): sel[c] = sel[c].clip(upper=6)
print(f"jump features: {len(jz_all)} → kept {len(jz_keep)} (P(|z|>{TAIL_Z:.0f}) > {TAIL_MIN:.1%}); total selected = {sel.shape[1]}")
print("dropped:", sorted(set(jz_all) - set(jz_keep)))
Z = sel.values.astype(np.float32)

# IF บนฟีเจอร์ทั้งหมด (แบบเดิม) เพื่อเทียบ
full = Fall.clip(upper=50).copy()
for k in (2, 3, 4): full[f"geo_disp{k}"] = np.log1p(home[:, k-1])
full["invalid_lat"] = sel["invalid_lat"]
for c in X.columns:
    if c[-2:] not in ("_1", "_2", "_3", "_4"): full[c] = X[c].values
full = full.fillna(0).values

# ---------- 2) detectors ----------
t0 = time.time()
S = {}
S["IF-all"] = -IsolationForest(n_estimators=500, max_samples=1024, random_state=SEED, n_jobs=-1).fit(full).score_samples(full)
S["IF-sel"] = -IsolationForest(n_estimators=500, max_samples=1024, random_state=SEED, n_jobs=-1).fit(Z).score_samples(Z)
nn = NearestNeighbors(n_neighbors=K_KNN + 1, n_jobs=-1).fit(Z)
dist, _ = nn.kneighbors(Z); S["kNN"] = dist[:, 1:].mean(1)
lof = LocalOutlierFactor(n_neighbors=K_LOF, n_jobs=-1).fit(Z); S["LOF"] = -lof.negative_outlier_factor_
tm = pd.read_csv(os.path.join(HERE, "preds", "task5_tmlp_scores.csv"))
assert (tm["id"].values == ids.values).all()
S["TMLP"] = tm["tmlp"].values
S["ENS-all4"] = np.mean([rank(S[k]) for k in ("IF-sel", "kNN", "LOF", "TMLP")], 0)   # ไว้เทียบใน report
S["ENS"] = np.mean([rank(S[k]) for k in ("IF-sel", "TMLP")], 0)   # เฉพาะตัวที่แข็งและเสริมกัน (Spearman ต่ำ)
print(f"detectors done in {time.time()-t0:.0f}s")

# ---------- 3) sanity (proxy ≠ ground truth) ----------
g3, g4 = np.nan_to_num(home[:, 2]) > 5, np.nan_to_num(home[:, 3]) > 5
inv = sel["invalid_lat"].values > 0
shock = np.stack([Fall[f"{b}_jz4"] > 6 for b in NONGEO], 1).sum(1)
proxy = g3 | g4 | inv | (shock >= 2)
types = {"VPN": g3, "theft": g4 & ~g3, "inv_lat": inv, "shock2": (shock >= 2) & ~g3 & ~g4}
print(f"\nproxy-obvious = {proxy.sum()} rows (sanity only)")
rows = []
for k, s in S.items():
    r = {"detector": k}
    for q in (0.032, 0.05):
        top = s >= np.quantile(s, 1 - q); r[f"recall@{q:.1%}"] = round((top & proxy).sum() / proxy.sum(), 3)
    top = s >= np.quantile(s, 0.95)
    r.update({t: f"{(top & m).sum()}/{m.sum()}" for t, m in types.items()})
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\nSpearman correlation:")
print(pd.DataFrame(S).corr(method="spearman").round(2).to_string())

# overlap ของส่วน "เกินกลุ่มชัดเจน" ที่ flag 5%
print("\noverlap beyond proxy @5% (Jaccard):")
ext = {k: (s >= np.quantile(s, .95)) & ~proxy for k, s in S.items()}
keys = list(ext)
print(pd.DataFrame([[round((ext[a] & ext[b]).sum() / max((ext[a] | ext[b]).sum(), 1), 2) for b in keys] for a in keys],
                   index=keys, columns=keys).to_string())

# ---------- 4) seed stability ของ IF-sel ----------
tops = []
for sd in range(5):
    s = -IsolationForest(n_estimators=500, max_samples=1024, random_state=sd, n_jobs=-1).fit(Z).score_samples(Z)
    tops.append(s >= np.quantile(s, .95))
jac = [ (tops[i] & tops[j]).sum() / (tops[i] | tops[j]).sum() for i in range(5) for j in range(i+1, 5)]
print(f"\nIF-sel seed stability (top 5%, Jaccard 5 seeds): mean {np.mean(jac):.3f}, min {np.min(jac):.3f}")

# ---------- 5) outputs ----------
os.makedirs(os.path.join(HERE, "preds"), exist_ok=True); os.makedirs(os.path.join(HERE, "probes"), exist_ok=True)
pd.DataFrame({"id": ids, **S}).to_csv(os.path.join(HERE, "preds", "task5_detectors_scores.csv"), index=False)
print("\nเขียน preds/task5_detectors_scores.csv แล้ว — ไม่ได้ submit อะไร")
