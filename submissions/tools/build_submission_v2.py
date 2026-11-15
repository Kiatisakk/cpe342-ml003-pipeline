"""สร้างไฟล์ submission ให้รูปแบบตรง sample_submission (comma, LF, ไม่มี BOM, ตัวเลขเป็น float เช่น 1.0 / 2983.2)
ใช้โมเดลที่ดีที่สุดที่มีในเครื่อง: ถ้ามีไฟล์ XGB จาก Kaggle ใน tX/preds/ จะใช้ก่อน ไม่งั้นใช้ HGB — ไม่ submit"""
import os, re, numpy as np, pandas as pd
S = pd.read_csv("cpe342-kaggle-main/sample_submission.csv"); N = len(S)
def pick(task):
    k = f"{task}/preds/{task}_xgb.csv"
    if os.path.exists(k):
        d = pd.read_csv(k); assert (d["id"].values == S["id"].values).all(); return d["pred"].values, "xgb"
    return pd.read_csv(f"{task}/preds/{task}_hgb.csv")["pred"].values, "hgb"
(t1, m1), (t2, m2), (t3, m3) = pick("t1"), pick("t2"), pick("t3")
t4 = pd.read_csv("t4/preds/v3a/task4_v3a_f0.csv")["pred_num"].values
e5 = pd.read_csv("t5/preds/task5_detectors_scores.csv"); assert (e5["id"].values == S["id"].values).all()
T5_N = 1000; t5 = np.zeros(N); t5[np.argsort(-e5["ENS"].values, kind="stable")[:T5_N]] = 1
sub = pd.DataFrame({"id": S["id"], "task1": t1.astype(float), "task2": t2.astype(float),
                    "task3": np.round(np.clip(t3.astype(float), 0, None), 2), "task4": t4.astype(float), "task5": t5})
tag = f"t1{m1}_t2{m2}_t3{m3}_t4v3a_t5ens{T5_N}"
fA = f"submissions/history/sub_v2_{tag}.csv"; sub.to_csv(fA, index=False, lineterminator="\n")
names = ["7-11k", "fiveN", "freefried", "pube", "roblock"]
subB = sub.copy(); subB["task4"] = [names[int(i)] for i in t4]
fB = f"submissions/history/sub_v2_{tag}_T4STRING.csv"; subB.to_csv(fB, index=False, lineterminator="\n")

# ---- ตรวจรูปแบบเทียบ sample ----
for f in (fA, fB):
    raw = open(f, "rb").read(); L = raw.decode().splitlines()
    ok = [L[0] == "id,task1,task2,task3,task4,task5", len(L) - 1 == N, raw[:3] != b"\xef\xbb\xbf", b"\r\n" not in raw,
          all(l.split(",")[0] == i for l, i in zip(L[1:], S["id"])), all(l.count(",") == 5 for l in L[1:]),
          not any(",," in l or l.endswith(",") for l in L[1:])]
    num = re.compile(r"^-?\d+\.\d+$")
    fmt = all(num.match(v) for l in L[1:] for j, v in enumerate(l.split(",")[1:]) if not (f == fB and j == 3))
    print(f, "| header/rows/BOM/LF/id-order/5 commas/no blanks:", ok, "| ตัวเลขเป็น x.y ทุกช่อง:", fmt)
    print("   ", L[1], "|", L[2])
print("t1 rate", sub.task1.mean().round(3), "| t2", sub.task2.value_counts(normalize=True).round(3).to_dict(),
      "| t3 zero", (sub.task3 == 0).mean().round(3), "| t4", sub.task4.value_counts(normalize=True).round(3).to_dict(), "| t5", int(sub.task5.sum()))
print("models:", m1, m2, m3)
