# แผน Masking: แยกคะแนน Public LB ราย Task

> TA อนุญาตแล้ว (2 พ.ย. 2026) ว่าเป็นการ masking ไม่ผิดกฎ
> **ยังไม่ได้ส่ง submission ใดๆ** ไฟล์ทั้งหมดสร้างในเครื่องเท่านั้น

## หลักการ

Kaggle แสดงแค่ `Score = Σ aₖ·sₖ` (aₖ = wₖ/Σw) ถ้าเรา "mask" task ที่ไม่สนใจด้วยค่าที่รู้คะแนนแน่นอน
แล้วเปิดโมเดลจริงทีละ task ส่วนต่างของคะแนนจะบอกคะแนนของ task นั้น

| Task | ค่า mask | คะแนนตอน mask | เหตุผล |
|---|---|---|---|
| 1 | `2` (label ที่ไม่มีจริง) | 0 | ไม่มี class 0/1 ถูกทายเลย → F2 ทุก class = 0 |
| 2 | `9` | 0 | เหมือนกัน |
| 3 | `0` | **0.5 แน่นอน** | NMAE = 1/(1+Σ\|y−0\|/Σy) = 1/2 |
| 4 | `9` | 0 | เหมือนกัน |
| 5 | `2` | 0 | เหมือนกัน |

## ลำดับการส่ง (ทั้งหมด 6 ครั้ง, โควตา 5/วัน)

| ครั้ง | ไฟล์ | ได้อะไร |
|---|---|---|
| P0 | `P0_all_mask.csv` | `a₃ = 2·P0` → ถ้า P0 = **0.1000** แปลว่า weight เท่ากันทุก task (0.2) และยืนยันว่า Kaggle รับ label ที่ไม่มีจริง |
| P1–P4 | `Pk_taskk_only_mask.csv` | คะแนน public ของ task 1–4 |
| P5 | `P5_task5_only_mask.csv` | คะแนน public ของ **task 5** ← คุ้มที่สุด เพราะไม่มี label ให้ทำ CV |

ถ้า weight เท่ากัน: `sₖ = 5·(Pk − P0)` (task 3 ต้อง +0.5)

**หลังจากนี้ส่งแค่ 1 ครั้งต่อการทดลอง 1 อย่าง** เช่นเปลี่ยน contamination ของ task 5
ส่งไฟล์ P5 เวอร์ชันใหม่ แล้วเทียบกับ P0 เดิม ไม่ต้องส่ง P0 ซ้ำ

## ถ้า P0 ส่งไม่ผ่าน (Kaggle error เพราะ label ไม่ถูกต้อง)

ใช้ `--fallback` (mask เป็น 0 ทุก task) แล้ว:
- คะแนน mask ของ task 3 ยังเป็น 0.5 แน่นอน แต่ task อื่นไม่เป็น 0
- ยังใช้ differential ได้: `Δₖ = aₖ·(sₖ − mₖ)` แต่ต้องคำนวณ mₖ จาก prevalence ใน train
- ความแม่นยำลดลง ให้พึ่ง local CV เป็นหลัก

## ข้อควรระวัง

1. **นี่คือคะแนน public LB เท่านั้น** ซึ่งเป็นแค่ส่วนหนึ่งของ test เกรดจริงคิดจาก private LB อย่า overfit public
2. **ความละเอียดของคะแนนที่ Kaggle แสดง** ถ้าแสดง 5 ตำแหน่ง × 5 (จาก a=0.2) จะคลาดเคลื่อนราย task ได้ ~±0.0001
3. **ไฟล์จริงตอนท้ายต้องเป็นโมเดลจริงครบ 5 task** อย่าลืมเลือก final submission ให้ถูกไฟล์
4. **ตัวคั่น:** หน้า Data Description บอกว่าใช้ `|` แต่ `sample_submission.csv` ที่ดาวน์โหลดมาใช้ `,` สคริปต์ใช้ตัวคั่นตามไฟล์ sample
5. **Task 4 เป็นตัวเลข 0–4** ใน sample (0: 7-11 Knights, 1: FiveN, 2: Free Fried, 3: PUBE, 4: Roblock) ไม่ใช่ชื่อเกมแบบในสไลด์

## วิธีใช้สคริปต์

```bash
cd masking
# 1) สร้างไฟล์ probe (ไฟล์ทำนายต้องมีคอลัมน์ id,pred)
python make_probes.py --sample ../cpe342-kaggle-main/sample_submission.csv \
    --pred 1=../preds/task1.csv --pred 5=../preds/task5.csv

# 2) ส่งไฟล์เอง (ด้วยมือ) แล้วกรอกคะแนนลง scores.json

# 3) แกะคะแนนราย task
python decode.py scores.json
```

ทดสอบกับข้อมูลจำลองแล้ว: แกะคะแนนกลับมาได้ตรงกับค่าจริงทุก task (sklearn macro F-beta)

---

## Task 5 — แผนส่ง probe รอบแรก (อัปเดต 2 พ.ย.)

| ลำดับ | ไฟล์ | ตอบคำถาม |
|---|---|---|
| ① | `P0_all_mask.csv` | ค่าอ้างอิง + weight |
| ② | `P5_task5_only_mask_ens_n830.csv` | flag เฉพาะกลุ่มชัดเจน 830 แถว → ใช้ `python task5_flag_rate.py <คะแนน task5>` ประมาณจำนวน anomaly จริง K |
| ③ | `P5_task5_only_mask_ens_c050.csv` | ถ้า ③ > ② = โมเดลจัดอันดับแถวก้ำกึ่งได้ดี และ K มากกว่า 830 |
| ④ | `P5_task5_only_mask_ifsel_c050.csv` หรือ `tmlp_only_c050` | IF กับ TMLP ตัวไหนเลือกแถวก้ำกึ่งถูกกว่า (overlap กันแค่ ~2%) |

ENS = rank average ของ IF-sel + TMLP (ไม่ใช้ kNN/LOF เพราะอ่อนกว่าและดึง ensemble ลง)
