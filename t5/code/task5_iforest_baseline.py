"""
Task 5 baseline: Isolation Forest (unsupervised) -> preds/task5_iforest_c{rate}.csv
ต้องติดตั้ง: pip install pandas scikit-learn
ใช้: python task5_iforest_baseline.py ../cpe342-kaggle-main/05_anomaly_detection/test.csv ../cpe342-kaggle-main/sample_submission.csv
"""
import sys, os
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

test_path, sample_path = sys.argv[1], sys.argv[2]
RATES = [0.05, 0.08, 0.10, 0.15]   # สัดส่วนที่จะ flag ว่าเป็น anomaly
SEED = 42

df = pd.read_csv(test_path)
ids = pd.read_csv(sample_path)["id"]
assert len(df) == len(ids), "จำนวนแถว test ไม่ตรงกับ sample_submission"

X = df.drop(columns=["player_id"])
# ฟีเจอร์เพิ่ม: ความผันผวนข้าม 4 ช่วงเวลา และการเปลี่ยนแปลงช่วงสุดท้าย
bases = sorted({c.rsplit("_", 1)[0] for c in X.columns if c[-2:] in ("_1", "_2", "_3", "_4")})
extra = {}
for b in bases:
    cols = [f"{b}_{i}" for i in range(1, 5)]
    v = X[cols]
    extra[f"{b}_std"] = v.std(axis=1)
    extra[f"{b}_delta"] = v[cols[3]] - v[cols[:3]].mean(axis=1)
X = pd.concat([X, pd.DataFrame(extra)], axis=1)
X = X.fillna(X.median())
X = RobustScaler().fit_transform(X)

model = IsolationForest(n_estimators=500, max_samples=1024, random_state=SEED, n_jobs=-1)
model.fit(X)
score = -model.score_samples(X)          # ยิ่งสูง ยิ่งผิดปกติ

os.makedirs("preds", exist_ok=True)
pd.DataFrame({"id": ids, "score": score}).to_csv("preds/task5_iforest_scores.csv", index=False)
for r in RATES:
    thr = np.quantile(score, 1 - r)
    pred = (score >= thr).astype(int)
    out = f"preds/task5_iforest_c{int(r*100):02d}.csv"
    pd.DataFrame({"id": ids, "pred": pred}).to_csv(out, index=False)
    print(f"{out}: flag {pred.sum()} แถว ({pred.mean():.1%})")
