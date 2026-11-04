"""รวมคำตอบที่ดีที่สุดของแต่ละ task เป็นไฟล์เดียว — ไม่ submit (ส่งเองที่หน้า Kaggle)"""
import numpy as np, pandas as pd, sys
S = pd.read_csv("cpe342-kaggle-main/sample_submission.csv"); N = len(S)
T5_N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000      # จำนวนแถวที่ flag ใน Task 5
t1 = pd.read_csv("t123/preds/t1_hgb.csv")["pred"].astype(int)
t2 = pd.read_csv("t123/preds/t2_hgb.csv")["pred"].astype(int)
t3 = pd.read_csv("t123/preds/t3_hgb.csv")["pred"].round(2)
t4 = pd.read_csv("task4/preds/task4_v3a_f0.csv")["pred_num"].astype(int)
e5 = pd.read_csv("masking/preds/task5_detectors_scores.csv")
assert (e5["id"].values == S["id"].values).all()
t5 = np.zeros(N, int); t5[np.argsort(-e5["ENS"].values, kind="stable")[:T5_N]] = 1
for name, col in (("t1", t1), ("t2", t2), ("t3", t3), ("t4", t4)): assert len(col) == N, name
sub = pd.DataFrame({"id": S["id"], "task1": t1, "task2": t2, "task3": t3, "task4": t4, "task5": t5})
out = f"submissions/sub_v1_t1hgb_t2hgb_t3hgb2stage_t4v3a_t5ens{T5_N}.csv"
sub.to_csv(out, index=False)
print(out, sub.shape); print(sub.head(3).to_string(index=False))
print("task1 rate", round(t1.mean(), 3), "| task2", t2.value_counts(normalize=True).round(3).to_dict())
print("task3 zero frac", round((t3 == 0).mean(), 3), "median", t3.median(), "| task4", t4.value_counts(normalize=True).round(3).to_dict(), "| task5 flagged", t5.sum())
print("NaN:", int(sub.isna().sum().sum()), "| ids unique:", sub.id.is_unique, "| same id order as sample:", (sub.id == S.id).all())
