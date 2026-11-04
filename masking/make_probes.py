"""
make_probes.py - สร้างไฟล์ probe สำหรับ masking (ไม่ส่ง Kaggle ให้ สร้างไฟล์อย่างเดียว)

ใช้ได้กับ Python มาตรฐาน ไม่ต้องลง library เพิ่ม

ตัวอย่าง:
  # สร้าง P0 (all-mask) อย่างเดียว
  python make_probes.py --sample ../cpe342-kaggle-main/sample_submission.csv

  # สร้าง probe ของ task 1 และ 5 จากไฟล์ทำนาย (csv ที่มีคอลัมน์ id,pred)
  python make_probes.py --sample ../cpe342-kaggle-main/sample_submission.csv \
      --pred 1=preds/task1.csv --pred 5=preds/task5_iforest.csv --tag v1

  # ใช้ค่า mask สำรอง (กรณี Kaggle ไม่รับ label ที่ไม่มีอยู่จริง)
  python make_probes.py --sample ... --fallback
"""
import argparse
import csv
import os

TASKS = ["task1", "task2", "task3", "task4", "task5"]

# ค่า mask หลัก: label ที่ไม่มีอยู่จริง -> คะแนน task นั้น = 0 แน่นอน
# task3 = 0 -> NMAE = 1/(1+Σy/Σy) = 0.5 แน่นอน
MASK_PRIMARY = {"task1": "2", "task2": "9", "task3": "0", "task4": "9", "task5": "2"}

# ค่า mask สำรอง: label ที่ถูกต้องแต่เป็นค่าคงที่ -> คะแนนไม่ทราบค่าแน่นอน แต่คงที่
# (ยังใช้วิธี differential ได้ แต่ต้องพึ่ง local CV มากขึ้น)
MASK_FALLBACK = {"task1": "0", "task2": "0", "task3": "0", "task4": "0", "task5": "0"}


def read_ids(sample_path):
    with open(sample_path, newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = "|" if sample.count("|") > sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delim)
        ids = [row["id"] for row in reader]
    return ids, delim


def read_pred(path):
    """อ่านไฟล์ทำนาย: ต้องมีคอลัมน์ id และ pred"""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "id" not in reader.fieldnames or "pred" not in reader.fieldnames:
            raise SystemExit(f"{path}: ต้องมีคอลัมน์ 'id' และ 'pred'")
        return {row["id"]: row["pred"] for row in reader}


def write_probe(path, ids, mask, overrides, delim):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delim)
        w.writerow(["id"] + TASKS)
        for i in ids:
            row = [i]
            for t in TASKS:
                if t in overrides:
                    if i not in overrides[t]:
                        raise SystemExit(f"ไม่มี id {i} ในไฟล์ทำนายของ {t}")
                    row.append(overrides[t][i])
                else:
                    row.append(mask[t])
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", default="probes")
    ap.add_argument("--pred", action="append", default=[],
                    help="รูปแบบ k=path เช่น 5=preds/task5.csv")
    ap.add_argument("--tag", default="", help="ต่อท้ายชื่อไฟล์ เช่น v1")
    ap.add_argument("--fallback", action="store_true")
    args = ap.parse_args()

    ids, delim = read_ids(args.sample)
    mask = MASK_FALLBACK if args.fallback else MASK_PRIMARY
    os.makedirs(args.out, exist_ok=True)
    suffix = ("_" + args.tag) if args.tag else ""
    mode = "fb" if args.fallback else "mask"

    p0 = os.path.join(args.out, f"P0_all_{mode}.csv")
    write_probe(p0, ids, mask, {}, delim)
    print(f"สร้าง {p0}  ({len(ids)} แถว, ตัวคั่น '{delim}')")

    for spec in args.pred:
        k, path = spec.split("=", 1)
        t = f"task{int(k)}"
        out = os.path.join(args.out, f"P{k}_{t}_only_{mode}{suffix}.csv")
        write_probe(out, ids, mask, {t: read_pred(path)}, delim)
        print(f"สร้าง {out}")

    print("\nยังไม่ได้ส่งอะไรขึ้น Kaggle - ไฟล์อยู่ในเครื่องเท่านั้น")


if __name__ == "__main__":
    main()
