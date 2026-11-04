"""
decode.py - แปลงคะแนน Public LB ของไฟล์ probe กลับเป็นคะแนนราย task

กรอกคะแนนที่ได้จาก Kaggle ลงใน scores.json แล้วรัน:
  python decode.py scores.json

ตัวอย่าง scores.json:
{
  "P0": 0.10000,                 # คะแนนของ P0_all_mask.csv
  "probes": {                    # คะแนนของไฟล์ที่เปิดโมเดลจริงทีละ task
    "1": 0.27000,
    "2": 0.25000,
    "3": 0.27000,
    "4": 0.28000,
    "5": 0.22000
  },
  "cv": {"1": 0.86, "2": 0.74, "4": 0.90},   # (ไม่บังคับ) คะแนน local CV ใช้กรณี weight ไม่เท่ากัน
  "mask_t3": 0.5                 # คะแนน task3 ตอน mask (ทาย 0 ทุกแถว) = 0.5
}

สูตร:  Score = Σ a_k·s_k   โดย a_k = w_k/Σw
  P0      = a3·0.5                         -> a3 = 2·P0
  probe_k = P0 + a_k·(s_k − m_k)           -> s_k = m_k + (probe_k − P0)/a_k
  (m_k = คะแนนตอน mask: 0 สำหรับ task 1,2,4,5 และ 0.5 สำหรับ task 3)
"""
import json
import sys

TOL = 0.002  # ความคลาดเคลื่อนที่ยอมรับเมื่อเช็กว่า weight เท่ากันหรือไม่


def main(path):
    d = json.load(open(path, encoding="utf-8"))
    p0 = d["P0"]
    m3 = d.get("mask_t3", 0.5)
    probes = {int(k): v for k, v in d.get("probes", {}).items()}
    cv = {int(k): v for k, v in d.get("cv", {}).items()}
    mask = {1: 0.0, 2: 0.0, 3: m3, 4: 0.0, 5: 0.0}

    a3 = p0 / m3
    print(f"P0 = {p0:.5f}  ->  a3 (weight ของ task3) = {a3:.4f}")

    equal = abs(a3 - 0.2) < TOL
    if equal:
        print("a3 ≈ 0.2  ->  สันนิษฐานว่า weight ทุก task เท่ากัน (0.2)")
        a = {k: 0.2 for k in range(1, 6)}
    else:
        print("a3 ≠ 0.2  ->  weight ไม่เท่ากัน ประมาณจาก local CV")
        a = {3: a3}
        for k in (1, 2, 4):
            if k in probes and k in cv and cv[k] > 0:
                a[k] = (probes[k] - p0) / cv[k]
        known = [1, 2, 3, 4]
        if all(k in a for k in known):
            a[5] = 1 - sum(a[k] for k in known)
        for k in sorted(a):
            print(f"  a{k} ≈ {a[k]:.4f}")

    print("\nคะแนน Public LB ราย task:")
    for k in range(1, 6):
        if k not in probes:
            print(f"  task{k}: (ยังไม่มี probe)")
            continue
        if k not in a or a[k] <= 0:
            print(f"  task{k}: หา weight ไม่ได้ (ต้องมี cv ของ task 1,2,4 ก่อน)")
            continue
        s = mask[k] + (probes[k] - p0) / a[k]
        note = "" if equal or k == 3 else "  (ประมาณ, ขึ้นกับ CV)"
        print(f"  task{k}: {s:.4f}{note}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("ใช้: python decode.py scores.json")
    main(sys.argv[1])
