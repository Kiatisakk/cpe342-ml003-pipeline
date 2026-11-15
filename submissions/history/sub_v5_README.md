# Submission v5 — `sub_v5_t1hgb_t2nb_t3blend_t4ovr_t5union.csv`

**Team:** ML003 · **Competition:** Player Intelligence System (CPE342 ML) · **สร้างเมื่อ:** 4 พ.ย. 2026

| Task | งาน | Metric | โมเดลใน v5 | CV / OOF |
|---|---|---|---|---|
| T1 | ตรวจจับผู้เล่นโกง (binary) | F2 macro | HistGradientBoosting + age-bound features + threshold tuning | F2 ≈ 0.781 |
| T2 | แบ่ง segment ผู้เล่น (4 คลาส) | F1 macro | Naive Bayes (Optuna per-feature likelihood) | F1 ≈ 0.758 |
| T3 | ทำนายยอดใช้จ่าย (regression, zero-inflated) | NMAE score | Hurdle XGB multi-quantile v4 + NN-ZILN blend | 0.7925 |
| T4 | จำแนกภาพเกม 5 คลาส | F1 macro | ConvNeXt-Tiny (DINOv3) fine-tune | F1 ≈ 0.99 |
| T5 | ตรวจจับบัญชีผิดปกติ (unsupervised) | F3 macro | Rule-based + stationary clusters (union 1,811 แถว) | ไม่มี label |

รูปแบบไฟล์: `id,task1..task5` เป็นตัวเลขแบบ float, คั่นด้วย comma, ขึ้นบรรทัดแบบ LF, 25,889 แถว

---

## Task 1 — Cheater Detection (F2 macro)

**ข้อค้นพบจาก EDA**
- มี **16 ฟีเจอร์ที่เป็น noise** (กระจายเท่ากันทุกคลาส) จึงเหลือ **15 ฟีเจอร์ที่มีสัญญาณ** เช่น kill_death_ratio, headshot_percentage, win_rate, accuracy_score, reaction_time_ms, reports_received, aiming_smoothness ฯลฯ
- มี **class-exclusive range zones**: ช่วงค่าบางช่วงของฟีเจอร์มีแต่ cheater หรือมีแต่คนปกติเท่านั้น ส่วนที่เหลือเป็นแกนกลางที่ซ้อนทับกันราว 55% (ส่วนนี้แยกได้แค่ AUC ≈ 0.79) จึงเป็นเพดานของงานนี้
- **Identity ที่ซ่อนอยู่:** `level_progression_speed = level · r / account_age_days` โดย r ~ U(0.15, 1.08)

**Feature engineering**
- ใช้ identity ข้างบนสร้างขอบเขตของค่าที่หาย:
  `age_lo = level·0.15/lps`, `age_hi = level·1.08/lps`, `age_filled = √(age_lo·age_hi)` และ `lv_lo`, `lv_hi` ในทำนองเดียวกัน
- ตัดฟีเจอร์ noise ออก เพราะทำให้ต้นไม้ split มั่ว

**โมเดล**
- `HistGradientBoostingClassifier` (lr 0.03, 300 iter, 31 leaves) จัดการค่า NaN ได้ในตัว
- **Threshold tuning:** F2 ให้น้ำหนัก recall มากกว่า precision จึงสแกน threshold บน OOF แล้วเลือกจุดที่ F2 macro สูงสุด (ได้ราว 0.39 ต่ำกว่า 0.5)

**ที่ลองแล้วแต่ไม่ได้ใช้ใน v5** (ผลใกล้เคียงกันหมด แปลว่าถึงเพดานของข้อมูลแล้ว)

| วิธี | F2 macro (OOF) |
|---|---|
| TabPFN v3.5 (tabpfn-client) | 0.7799 |
| TabPFN + HGB average | 0.7825 |
| EBM / HGB + EBM blend | 0.7786 / 0.7812 |
| XGB + LGBM + CatBoost, Optuna 200 trials, weight blend | 0.7804 |

---

## Task 2 — Player Segmentation (F1 macro)

**ข้อค้นพบจาก EDA**
- ภายในแต่ละคลาส ฟีเจอร์**เป็นอิสระต่อกันแบบมีเงื่อนไข** (correlation ภายในคลาส ≈ 0) แสดงว่าข้อมูลน่าจะถูกสร้างแบบ class-conditional independent ซึ่งทำให้ **Naive Bayes เป็นตัวจำแนกที่ Bayes-optimal**
- โครงสร้างแฝงเป็นแบบ 2×2 (engagement × spending)
- categorical ที่มีสัญญาณมีแค่ `loyalty_tier` กับ `engagement_level` ตัวอื่นเป็น noise

**โมเดล**
- Naive Bayes แบบผสม: เลือกการแจกแจงของแต่ละฟีเจอร์แยกกัน (Gaussian / log-Gaussian / KDE / histogram) ส่วน categorical ใช้ count + Laplace smoothing
- **Optuna** จูนราย feature: ชนิด likelihood, bandwidth/bins, smoothing, การเลือกฟีเจอร์ และ class prior adjustment เพื่อให้ F1 macro สูงสุด
- ค่าที่หายตัดออกจาก likelihood (marginalize) แทนการ impute ซึ่งเป็นวิธีที่ถูกต้องสำหรับ NB

**ผล:** F1 macro ≈ 0.758 ดีกว่า HGB (≈ 0.750) และ XGB pipeline (0.7498)

---

## Task 3 — Spending Prediction (NMAE score, ยิ่งสูงยิ่งดี)

`score = 1 / (1 + MAE / mean|y|)` ถ้าทายเป็น 0 ทั้งหมดจะได้ 0.5

**ลักษณะข้อมูล:** ~48% ของ target เป็น 0 (zero-inflated) และมี whale (segment 3) ที่ทำให้เกิด **~80% ของ error ทั้งหมด**

**Preprocessing / Imputation**
- **Exact identities** เติมค่าที่หายได้แบบแม่นยำ:
  - `historical_spending = total_transactions × avg_transaction_value`
  - `purchases_on_discount = total_transactions × discount_rate_used`
  ทำให้ค่าที่หายลดจาก ~5.8% เหลือ ~0.4%
- **Segment imputer:** ทำนาย segment ที่หายไปด้วยโมเดลจากฟีเจอร์อื่น ได้ holdout acc 0.929

**โมเดล (Hurdle / ZILN)**
- **Stage 1:** classifier ทำนาย p = P(y > 0)
- **Stage 2:** โมเดลจำนวนเงินเมื่อ y > 0
  - **XGB multi-quantile** (`reg:quantileerror`): ค่าที่ทำให้ MAE ต่ำสุดของ hurdle mixture คือ quantile **q = 1 − 1/(2p)** ของ conditional distribution ถ้า p < 0.5 ให้ทาย 0
  - ใช้ทั้ง quantile แบบไม่ถ่วงและแบบถ่วงน้ำหนัก (qw0.3, qw0.6) แล้วปรับ calibration (shift, κ) บน OOF
- **NN-ZILN:** MLP ที่ output เป็น zero-probability + mixture of K=3 lognormals ฝึกด้วย ZILN negative log-likelihood แล้วดึง quantile ที่ MAE-optimal จาก mixture
- XGB 4,000–15,000 ต้น, lr 0.003, early stopping 100, 5-fold

**Blend (หาน้ำหนักบน OOF 104k แถว)**

| โมเดล | น้ำหนัก | NMAE เดี่ยว |
|---|---|---|
| v4 D_q | 0.38 | 0.7917 |
| v4 D_qw0.3 | 0.24 | 0.7914 |
| v4 D_qw0.6 | 0.06 | 0.7895 |
| NN-ZILN K3 | 0.32 | 0.7901 |
| **Blend** | | **0.7925** |

(HGB two-stage ที่ใช้ใน v2: 0.7843)

---

## Task 4 — Game Image Classification (F1 macro)

**โมเดล:** `convnext_tiny.dinov3_lvd1689m` (timm) ใช้ภาพขนาด 120×230
- **LP-FT:** linear probe ก่อน แล้วค่อย fine-tune ทั้งโมเดล
- **LLRD 0.8** (layer-wise LR decay) ให้ layer ตื้นเรียนช้ากว่า เพื่อรักษา feature จาก pretrain
- **EMA** ของน้ำหนัก, label smoothing, augmentation แบบเบา
- v5 ใช้ v3a (fold 0, F1 0.9915) ส่วน 5-fold ensemble (OOF 0.9889) รอใส่ในเวอร์ชันถัดไป

**⚠️ ลำดับคลาส (สำคัญ)**
- Datacard เรียงตามตัวอักษร: 0 7-11k, 1 fiveN, 2 freefried, 3 pube, 4 roblock
- **Kaggle Overview:** **0 PUBE Mobile, 1 Free Fried, 2 FiveN, 3 Roblock, 4 7-Eleven Knight**
- v5 ใช้ **ลำดับใน Overview** เพราะ LB 0.549 ของ v2 (ที่ใช้ลำดับ datacard) อธิบายได้ด้วยกรณีที่ T4 ได้ ~0 และการที่เพื่อนส่ง T4 แบบค่าคงที่กับแบบโมเดลแล้วคะแนนต่างกันแค่ ~0.03 ก็สอดคล้องกับ mapping ผิดทุกคลาส ต้องยืนยันด้วย probe `sub_v2b_T4overview_order.csv`

---

## Task 5 — Account Anomaly Detection (unsupervised, F3 macro)

ข้อมูลมี 4 ช่วงเวลา × ~29 metric + พิกัด login + static profile และไม่มี label

**คณิตของ metric:** สำหรับคลาส anomaly, F3 = 10·tp / (9K + n) (K = จำนวน anomaly จริง, n = จำนวนที่ทาย) พลาด anomaly 1 แถวเสียคะแนนมากกว่าทายเกิน 1 แถวราว 9 เท่า จึงควร**ทายกว้างไว้ก่อน**

**กฎจากคำใบ้ของโจทย์ (827 แถว)**
- **Geo jump:** พิกัด login ช่วง 3/4 ห่างจาก "home" (ค่าเฉลี่ยช่วง 1–2) มากกว่า 5°
- **Invalid latitude:** |lat| > 90
- **Behaviour shock:** มี metric ≥ 2 ตัวที่เปลี่ยนจากช่วงก่อนด้วย robust z (median/MAD) > 6

**Stationary clusters (+984 แถว)**: ใช้ histogram-gap scan หากลุ่มที่แยกตัวชัดเจน
- BOT: mean `login_time_variance` < 1.0 (เล่นเวลาเดิมเป๊ะ ๆ)
- LOWACT: mean `activity_level` < 0.34
- LOWCOMM: mean `communication_frequency` < 0.17
- LOWSTAB: `behavior_stability` < 0.55

**v5 = union ของทั้งหมด = 1,811 แถว** ซึ่งตรงกับคำใบ้ "actual number likely 2x higher" (827 → ~1.8k)

**ที่ลองแล้ว:** Temporal Masked Autoencoder (Transformer, Huber loss บน masked cells, robust trimming) ได้ AUC 0.83 เทียบกับกฎ แต่จับกลุ่ม stationary ได้แค่ ~5% (AUC เทียบ union 0.66) เพราะพฤติกรรมนิ่ง ๆ reconstruct ได้ง่าย ซึ่งเป็นจุดอ่อนของ reconstruction-based detection จึงไม่ได้ใช้

---

## ประวัติ submission ที่เกี่ยวข้อง

| ไฟล์ | สิ่งที่เปลี่ยน | Public LB |
|---|---|---|
| sub_v2 | T1 HGB, T2 HGB, T3 HGB, T4 v3a (ลำดับ datacard), T5 ens top-1000 | 0.54905 |
| sub_v2b | sub_v2 + T4 ลำดับ Overview | รอ |
| sub_v3 | sub_v2b + T5 union 1811 | รอ |
| **sub_v5** | sub_v3 + T2 NB + T3 blend | รอ |

## งานถัดไป
- T4: ใส่ 5-fold ensemble (`task4_v3_5f_test_proba.npy`) แปลงเป็นลำดับ Overview
- T1: พิจารณา TabPFN + HGB average (0.7825)
- T5: ใช้ probe T5a / T5b / T5c ประมาณค่า K จริงจาก LB แล้วปรับ cutoff
