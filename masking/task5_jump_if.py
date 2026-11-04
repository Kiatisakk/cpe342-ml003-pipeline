"""
Task 5 v2: Isolation Forest บนฟีเจอร์ "การกระโดดข้ามช่วงเวลา" (จาก EDA: anomaly ถูกฉีดในช่วง _3/_4)
ต้องติดตั้ง: pip install pandas scikit-learn
ใช้: python task5_jump_if.py <test.csv> <sample_submission.csv>
"""
import sys, os
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest

test_path, sample_path = sys.argv[1], sys.argv[2]
RATES = [0.035, 0.05, 0.08]
X = pd.read_csv(test_path).drop(columns=["player_id"])
ids = pd.read_csv(sample_path)["id"]
assert len(X) == len(ids)

def haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

F = {}
bases = sorted({c.rsplit("_", 1)[0] for c in X.columns if c[-2:] in ("_1", "_2", "_3", "_4")})
for b in bases:
    c = [f"{b}_{i}" for i in range(1, 5)]
    for k in (3, 4):  # ช่วง k เทียบกับค่าเฉลี่ยช่วงก่อนหน้า -> robust z
        j = X[c[k-1]] - X[c[:k-1]].mean(axis=1)
        m = j.median(); s = 1.4826 * (j - m).abs().median() + 1e-9
        F[f"{b}_jz{k}"] = ((j - m) / s).abs()
for a, b in ((1, 2), (2, 3), (3, 4)):  # ระยะทางที่ตำแหน่งล็อกอินย้าย (km)
    F[f"geo_{a}{b}"] = np.log1p(haversine(X[f"login_lat_{a}"], X[f"login_lon_{a}"],
                                          X[f"login_lat_{b}"], X[f"login_lon_{b}"]))
for f in ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]:
    F[f] = X[f]
Z = pd.DataFrame(F).fillna(0).clip(upper=50)

model = IsolationForest(n_estimators=500, max_samples=1024, random_state=42, n_jobs=-1).fit(Z)
score = -model.score_samples(Z)
os.makedirs("preds", exist_ok=True)
pd.DataFrame({"id": ids, "score": score}).to_csv("preds/task5_jumpif_scores.csv", index=False)
for r in RATES:
    pred = (score >= np.quantile(score, 1 - r)).astype(int)
    out = f"preds/task5_jumpif_c{r*1000:03.0f}.csv"
    pd.DataFrame({"id": ids, "pred": pred}).to_csv(out, index=False)
    print(f"{out}: flag {pred.sum()} ({pred.mean():.1%})")
