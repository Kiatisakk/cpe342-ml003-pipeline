import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbtools import build, md, code, SETUP

C = []
C += [md(r'''
# Task 1 — Cheater Detection (Binary Classification)
**ทีม ML003 · Player Intelligence System (CPE342)**

Notebook นี้เล่ากระบวนการทำ Task 1 แบบเป็นขั้นเป็นตอน ตั้งแต่ดูข้อมูลครั้งแรก → ค้นพบโครงสร้างที่ซ่อนอยู่ (`level_progression_speed`) → สร้าง feature → โมเดล → การพิสูจน์ metric จริงบน leaderboard → โมเดลสุดท้าย
ทุกตัวเลขในตารางสรุปมาจากการทดลองจริงใน `t1/research/` (ไฟล์ OOF ที่บันทึกไว้) ส่วนโค้ดที่ train ใหม่ใน notebook นี้เป็นเวอร์ชันย่อ (learning rate สูงกว่า) เพื่อให้รันได้เร็ว

| หัวข้อ | เนื้อหา |
|---|---|
| 1 | โจทย์และ metric |
| 2 | ดูข้อมูล: ขนาด, label, ค่าที่หาย |
| 3 | แยก feature ที่มีสัญญาณ vs noise |
| 4 | โซนที่มีแต่คลาสเดียว (class-exclusive zones) |
| 5 | ค้นพบสูตรซ่อน `level_progression_speed = level · r / account_age_days` |
| 6 | ค่าที่หายเป็นแบบสุ่ม (MCAR) |
| 7 | Feature engineering (`fe_v2`) |
| 8 | Baseline vs fe_v2 |
| 9 | MCAR masking augmentation |
| 10 | Metric จริงบน LB = binary F2 และการเลือกจุดตัด |
| 11 | Rank blend สุดท้าย |
| 12 | สิ่งที่ลองแล้วไม่ได้ผล + เพดานของข้อมูล |
| 13 | สรุป |
'''), code(SETUP + r'''
sys.path.insert(0, "t1/research")
import common
common.DATA = os.path.join("cpe342-kaggle-main", "01_binary_classification")
from common import SIGNAL, TARGET, fe_v2, fe_baseline, cv_lgb, best_thr, nested_thr_score, folds_for
''')]

C += [md(r'''
## 1. โจทย์และ metric
- **Input:** สถิติการเล่นของผู้เล่น 31 คอลัมน์ (KD, headshot %, reaction time, account age, level, ...)
- **Output:** `is_cheater` ∈ {0, 1}
- **Metric บน leaderboard:** **F2-score ของคลาส 1 (binary F2)**

$$F_2 = \frac{5 \cdot TP}{5 \cdot TP + 4 \cdot FN + FP}$$

F2 ให้น้ำหนัก **recall มากกว่า precision 4 เท่า** (β² = 4) → การปล่อยคนโกงหลุด (FN) แพงกว่าการจับผิดคน (FP) มาก
ตอนเริ่มต้นเราเข้าใจว่าเป็น **F2 macro** (เฉลี่ยของ 2 คลาส) — ข้อ 10 จะแสดงว่าเราพิสูจน์ได้อย่างไรว่าที่จริงเป็น binary F2

**การวัดผลภายใน:** 5-fold StratifiedKFold (seed 42, และ 7 / 2024 เพื่อดูความแปรปรวน) + **nested threshold** = จุดตัดเลือกจาก 4 fold อื่นแล้วนำไปใช้กับ fold ที่เหลือ (กันการ overfit จุดตัด)
'''), code(r'''
def f2_binary(y, pred):
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    return 5 * tp / (5 * tp + 4 * fn + fp)

# ตัวอย่าง: recall สำคัญกว่า precision
y_demo = np.array([1]*100 + [0]*100)
a = np.r_[np.ones(95), np.zeros(5), np.ones(60), np.zeros(40)]   # recall .95, precision .61
b = np.r_[np.ones(70), np.zeros(30), np.ones(10), np.zeros(90)]  # recall .70, precision .88
print("recall สูง precision ต่ำ  F2 =", round(f2_binary(y_demo, a), 4))
print("recall ต่ำ precision สูง F2 =", round(f2_binary(y_demo, b), 4))
''')]

C += [md(r'''
## 2. ดูข้อมูล
'''), code(r'''
tr = pd.read_csv("cpe342-kaggle-main/01_binary_classification/train.csv")
te = pd.read_csv("cpe342-kaggle-main/01_binary_classification/test.csv")
lab, unl = tr[tr[TARGET].notna()].reset_index(drop=True), tr[tr[TARGET].isna()].reset_index(drop=True)
y = lab[TARGET].astype(int).values
print("train ทั้งหมด", tr.shape, "| มี label", len(lab), "| ไม่มี label", len(unl), "| test", te.shape)
print("สัดส่วนคนโกง (base rate) =", round(y.mean(), 4))
raw = [c for c in tr.columns if c not in ("player_id", TARGET)]
print("จำนวน feature ดิบ:", len(raw))
lab[raw].describe().T.round(3)
'''), code(r'''
miss = pd.DataFrame({"train": lab[raw].isna().mean(), "test": te[raw].isna().mean()}).sort_values("train", ascending=False)
ax = miss.plot.bar(figsize=(13, 3.5), title="Missing rate per column (train labeled vs test)"); ax.set_ylabel("fraction missing"); plt.tight_layout(); plt.show()
print("แถวที่ไม่มีค่าหายเลย: train %.1f%% | test %.1f%%" % (100 * lab[raw].notna().all(1).mean(), 100 * te[raw].notna().all(1).mean()))
''')]

C += [md(r'''
**สังเกต:** ทุกคอลัมน์มีค่าหายราว 5–15% และสัดส่วนใน train กับ test ใกล้กัน → ต้องออกแบบโมเดลให้ทนต่อค่าที่หาย (ข้อ 6, 9)
และมี **13,184 แถวใน train ที่ไม่มี label** → เก็บไว้ทดลอง semi-supervised (ข้อ 12)

## 3. Feature ไหนมีสัญญาณ, ไหนเป็น noise
วิธี: วัดแต่ละ feature เดี่ยว ๆ ว่าแยกคลาสได้ไหม
- **AUC เดี่ยว** (ใช้ค่า feature เป็นคะแนนตรง ๆ) → |AUC − 0.5| ใกล้ 0 = ไม่มีสัญญาณ
- **KS test** ระหว่างการกระจายของคลาส 0 กับ 1 → p-value สูง = การกระจายเหมือนกัน
'''), code(r'''
rows = []
for c in raw:
    v = lab[c]; m = v.notna()
    auc = roc_auc_score(y[m], v[m]); ks = ks_2samp(v[m & (y == 1)], v[m & (y == 0)])
    rows.append((c, abs(auc - .5), ks.statistic, ks.pvalue, c in SIGNAL))
sig = pd.DataFrame(rows, columns=["feature", "|AUC-0.5|", "KS stat", "KS p", "signal"]).sort_values("|AUC-0.5|", ascending=False)
fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(sig.feature, sig["|AUC-0.5|"], color=np.where(sig.signal, "tab:blue", "tab:gray"))
ax.set_title("Univariate separation |AUC-0.5| (blue = kept signal, gray = noise)"); plt.xticks(rotation=90); plt.tight_layout(); plt.show()
sig.round(4)
''')]

C += [md(r'''
**ผล:** 15 คอลัมน์มีสัญญาณชัด และ **16 คอลัมน์มี |AUC−0.5| ≈ 0 และ KS p > 0.07** = ไม่มีข้อมูลเกี่ยวกับคลาส
- บางคอลัมน์ noise ยังสัมพันธ์กับคอลัมน์สัญญาณ (เช่น input_consistency–reaction_time ρ≈0.26) แต่สัมพันธ์เท่ากันทั้งสองคลาส → ไม่ช่วยแยกคลาส
- ทดลองจริง: ใส่ 16 คอลัมน์นี้กลับเข้าไปทำให้ AUC ลด 0.0004 → **ตัดทิ้ง**

## 4. โซนที่มีแต่คลาสเดียว (class-exclusive zones)
ดูอัตราคนโกงตามช่วงค่าของ feature สำคัญ แล้วพบว่าหลายช่วง **มีแต่คนโกงหรือมีแต่คนปกติ 100%**
'''), code(r'''
fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
for ax, c, bins in [(axs[0], "account_age_days", np.arange(0, 1300, 10)), (axs[1], "level", np.arange(0, 101, 2)), (axs[2], "reports_received", np.arange(-0.5, 31, 1))]:
    d = lab[[c]].assign(y=y).dropna(); d["bin"] = pd.cut(d[c], bins)
    g = d.groupby("bin", observed=True).y.agg(["mean", "size"]); g = g[g["size"] >= 20]
    ax.plot([i.mid for i in g.index], g["mean"], ".-"); ax.axhline(y.mean(), color="r", ls="--", lw=1)
    ax.set_title(f"cheater rate by {c}"); ax.set_ylim(-.03, 1.03)
plt.tight_layout(); plt.show()
'''), code(r'''
age = lab.account_age_days
zones = {"age < 15 (legit-only)": age < 15, "age 110-120 (cheater-only)": age.between(110, 120, inclusive="left"),
         "age 180-200 (legit-only)": age.between(180, 200, inclusive="left"), "age >= 1000 (legit-only)": age >= 1000,
         "level < 28 (cheater-only)": lab.level < 28, "level > 96 (legit-only)": lab.level > 96,
         "reports == 4 (cheater-only)": lab.reports_received == 4, "reports >= 20 (legit-only)": lab.reports_received >= 20,
         "crosshair < 0.82 (cheater-only)": lab.crosshair_placement < 0.82, "crosshair > 0.94 (legit-only)": lab.crosshair_placement > 0.94}
pd.DataFrame({z: {"rows": int(m.sum()), "cheater rate": round(y[m.values].mean(), 4)} for z, m in zones.items()}).T
''')]

C += [md(r'''
**ข้อค้นพบ**
- คนปกติเป็นส่วนผสมของหลายกลุ่มตามอายุบัญชี (บัญชีใหม่ < 110 วันเพื่อนน้อย/โดนรีพอร์ตเยอะ, บัญชี 120–1000 วันเพื่อน 50–240) ส่วนคนโกงอยู่ช่วงอายุ 15–180 และ 200–1000 วัน → เกิด **หน้าต่างอายุที่มีคลาสเดียว**
- ประมาณ **34% ของแถว** อยู่ในโซนที่ตอบได้ 100% ที่เหลือ ~55–66% อยู่ใน **"แกนกลาง" ที่ค่าซ้อนทับกัน** (AUC ภายในแกน ≈ 0.77–0.79)
- → นี่คือ **เพดานของข้อมูล**: เปลี่ยนโมเดลอย่างเดียวจะไม่ได้เพิ่มมาก สิ่งที่ช่วยได้คือ "ทำให้แถวที่ค่าหายรู้ว่าตัวเองอยู่โซนไหน"

ปัญหา: ถ้า `account_age_days` หาย (≈14%) โมเดลจะไม่รู้ว่าอยู่ในหน้าต่างอายุไหน → ข้อ 5 แก้ปัญหานี้

## 5. ค้นพบสูตรซ่อน: `level_progression_speed`
สมมติฐาน: ความเร็วในการขึ้นเลเวล = เลเวล / อายุบัญชี อาจคูณด้วยตัวแปรสุ่มบางอย่าง
ทดสอบโดยคำนวณ **r = level_progression_speed × account_age_days / level**
'''), code(r'''
d = lab[["level", "account_age_days", "level_progression_speed"]].dropna()
r = d.level_progression_speed * d.account_age_days / d.level
print("r: min %.4f  max %.4f  mean %.4f  (ทฤษฎี U(0.15,1.08) mean = %.4f)" % (r.min(), r.max(), r.mean(), (0.15 + 1.08) / 2))
fig, axs = plt.subplots(1, 2, figsize=(13, 3.6))
axs[0].hist(r, 120, color="tab:blue"); axs[0].set_title("r = lps * age / level  -> flat between 0.15 and 1.08")
yy = y[d.index]
axs[1].hist([r[yy == 0], r[yy == 1]], 60, density=True, label=["legit", "cheater"], histtype="step", lw=2); axs[1].legend(); axs[1].set_title("same distribution in both classes")
plt.tight_layout(); plt.show()
print("AUC ของ r เดี่ยว ๆ =", round(roc_auc_score(yy, r), 4), "→ r ไม่มีข้อมูลคลาส แต่ทำให้ 'ย้อนหา' ค่าที่หายได้")
''')]

C += [md(r'''
**ผล:** r กระจาย **สม่ำเสมอ (uniform) ระหว่าง 0.15 ถึง 1.08** เท่ากันทั้งสองคลาส → ข้อมูลถูกสร้างด้วยสูตร

$$\text{level\_progression\_speed} = \frac{\text{level} \cdot r}{\text{account\_age\_days}}, \quad r \sim U(0.15,\ 1.08)$$

**ใช้ประโยชน์อย่างไร** เมื่อค่าใดค่าหนึ่งหาย แต่อีกสองค่ายังอยู่ เราย้อนหา **ช่วงที่เป็นไปได้** ได้ทันที
- อายุหาย: `age ∈ [level·0.15/lps, level·1.08/lps]` → feature `age_lo`, `age_hi`
- เลเวลหาย: `level ∈ [lps·age/1.08, lps·age/0.15]` → feature `lv_lo`, `lv_hi`

ดีกว่านั้นคือคำนวณ **posterior** ของอายุบนกริด 1 วัน: P(age) ∝ prior(age) · age · 1[r อยู่ใน 0.15–1.08] (ตัว age มาจาก Jacobian ของการแปลงตัวแปร)
แล้วได้ **ความน่าจะเป็นที่แถวนี้อยู่ในแต่ละโซนคลาสเดียว** (เช่น P(110 ≤ age < 120)) → feature `pa_*`, `pl_*` (= `fe_zone_post`)
prior ประมาณจากแถว train (ทั้งมีและไม่มี label) เท่านั้น ไม่ใช้ label และไม่ fit บน test
'''), code(r'''
ref = pd.concat([lab, unl])
Xz = common.fe_zone_post(lab, ref)
m_age = lab.account_age_days.isna() & lab.level.notna() & lab.level_progression_speed.notna()
print("แถวที่อายุหายแต่ย้อนหาได้:", int(m_age.sum()))
show = lab.loc[m_age, ["level", "level_progression_speed"]].head(5).copy()
show["age_lo"] = show.level * .15 / show.level_progression_speed; show["age_hi"] = show.level * 1.08 / show.level_progression_speed
show = show.join(Xz.loc[show.index, ["age_post_mean"] + [c for c in Xz.columns if c.startswith("pa_")]])
show.round(3)
'''), code(r'''
# posterior ช่วยได้จริงไหม: ในแถวที่อายุหาย, P(อยู่โซนคนโกง 110-120) สัมพันธ์กับ label หรือไม่
pc = Xz.loc[m_age, "p_cheat_zone"]; pl = Xz.loc[m_age, "p_legit_zone"]; ym = y[m_age.values]
for name, v in [("p_cheat_zone", pc), ("p_legit_zone", pl)]:
    b = pd.qcut(v.rank(method="first"), 5, labels=False)
    print(name, "→ cheater rate ต่อ quintile:", np.round(pd.Series(ym).groupby(b.values).mean().values, 3))
''')]

C += [md(r'''
## 6. ค่าที่หายเป็นแบบสุ่มจริง (MCAR)
ถ้าค่าที่หายเกิดแบบสุ่ม (Missing Completely At Random) อัตราคนโกงในแถวที่ค่าหายต้องเท่ากับ base rate
'''), code(r'''
mc = pd.DataFrame({c: {"missing rate": lab[c].isna().mean(), "cheater rate | missing": y[lab[c].isna().values].mean()} for c in SIGNAL}).T
mc["base rate"] = y.mean(); mc["diff"] = mc["cheater rate | missing"] - mc["base rate"]
mc.round(4)
''')]

C += [md(r'''
**ผล:** อัตราคนโกงเมื่อค่าหาย ≈ base rate ทุกคอลัมน์ → **การหายไม่ได้บอกคลาส**
ผลสืบเนื่องสองข้อ:
1. ไม่ต้องหาความหมายของ "การหาย" — แค่ต้องให้โมเดลทายได้ดีเมื่อข้อมูลไม่ครบ
2. **เราสามารถ "ลบค่าเพิ่ม" ในข้อมูลเทรนแบบสุ่มได้โดยไม่ทำให้ label ผิด** → เป็น data augmentation (ข้อ 9)

## 7. Feature engineering (`fe_v2`)
รวมทุกอย่างที่ค้นพบเป็นชุด feature เดียว:

| กลุ่ม | feature | ที่มา |
|---|---|---|
| สัญญาณดิบ | 15 คอลัมน์ (ตัด noise 16) | ข้อ 3 |
| flag ค่าหาย | `na_*`, `n_missing` | ข้อ 6 |
| ขอบเขตจากสูตรซ่อน | `age_lo/hi`, `age_filled`, `lv_lo/hi`, `lvl_filled`, `r_hat` | ข้อ 5 |
| โซนคลาสเดียว | `zone_cheat`, `zone_legit` | ข้อ 4 |
| posterior ของโซน | `pa_*`, `pl_*`, `p_cheat_zone`, `p_legit_zone`, `age_post_mean` | ข้อ 5 |
| อื่น ๆ | log, อัตราต่อ 100 วัน, skill z-score, frequency encoding | baseline pipeline |
'''), code(r'''
X = fe_v2(lab, ref); Xraw = lab[SIGNAL]
print("fe_v2:", X.shape[1], "features")
print(list(X.columns))
''')]

C += [md(r'''
## 8. Baseline vs fe_v2 (LightGBM)
ใช้ LightGBM พารามิเตอร์ที่จูนด้วย Optuna (200 trials) จาก baseline pipeline · 5-fold · early stopping บน 10% ภายใน fold
(ใน notebook ใช้ lr 0.1 เพื่อความเร็ว; การทดลองจริงใช้ lr 0.05 / 0.02 ผลอยู่ในตารางด้านล่าง)
'''), code(r'''
def nested_f2b(p, seed=42): return nested_thr_score(y, p, folds_for(y, seed), "binary")
res = {}
for name, XX in [("raw 15 signal cols", Xraw), ("fe_v2", X)]:
    oof, _ = cv_lgb(XX, y, seed=42, lr=0.1)
    res[name] = {"AUC": roc_auc_score(y, oof), "binary F2 (nested)": nested_f2b(oof)}
pd.DataFrame(res).T.round(5)
'''), md(r'''
ผลจากการทดลองจริง (lr 0.05, 3 seeds, จาก `t1/research/REPORT.md`):

| การทดลอง | AUC | F2 macro (nested) | binary F2 |
|---|---|---|---|
| raw 15 คอลัมน์ | 0.89166 | 0.77903 | 0.83687 |
| raw 31 คอลัมน์ (รวม noise) | 0.89129 | 0.77998 | 0.83709 |
| baseline FE | 0.89244 | 0.78006 | 0.83776 |
| **fe_v2 (+ zone posterior)** | **0.89315** | **0.78108** | 0.83840 |
| fe_v2, lr 0.02 | 0.89374 | 0.78151 | 0.83890 |
| XGBoost / CatBoost บน fe_v2 | 0.89279 / 0.89235 | 0.78054 / 0.77925 | 0.83864 / 0.83806 |
| MLP (PyTorch) / Mixture Naive Bayes | 0.88548 / 0.86957 | 0.77177 / 0.76215 | 0.83473 / 0.82083 |

**สรุป:** LightGBM ดีที่สุด, โมเดลตระกูล GBM ได้ใกล้กันมาก → ต่อไปเน้นที่ข้อมูล ไม่ใช่ที่โมเดล

## 9. MCAR masking augmentation
**Error analysis:** AUC = 0.918 เมื่อไม่มีค่าหาย แต่ลดเหลือ ~0.81 เมื่อหาย 5 ค่า และแถวที่ขาด reports/friends/age/crosshair อยู่ที่ 0.854–0.863
มีแถวที่ข้อมูลไม่ครบจริงแค่ ~16k แถวให้โมเดลเรียน

**ไอเดีย** (ใช้ได้เพราะข้อ 6): ในแต่ละ fold ทำสำเนาแถวเทรน 4 ชุด แล้ว **ซ่อนค่าแต่ละช่องด้วยความน่าจะเป็นเท่ากับอัตราการหายจริงของคอลัมน์นั้น** จากนั้นคำนวณ `fe_v2` ใหม่ (รวม posterior)
→ โมเดลเห็นตัวอย่าง "ข้อมูลไม่ครบ" จาก 72k แถวแทน 16k แถว
ห้าม augment: early-stopping holdout, fold ที่ใช้วัดผล, และ test
'''), code(r'''
miss_rate = lab[SIGNAL].isna().mean().values
def mask_copy(df, rng):
    d = df.copy(); M = rng.random((len(d), len(SIGNAL))) < miss_rate[None, :]
    v = d[SIGNAL].values.copy(); v[M] = np.nan; d[SIGNAL] = v
    return fe_v2(d, ref)
rng = np.random.default_rng(0); demo = mask_copy(lab.head(8), rng)
print("ตัวอย่าง: ค่าที่หายก่อน/หลัง mask ใน 8 แถวแรก =", int(lab.head(8)[SIGNAL].isna().sum().sum()), "->", int(demo[SIGNAL].isna().sum().sum()))
'''), code(r'''
# เทียบโมเดลจริงที่บันทึกไว้: round-1 (ไม่มี augmentation) vs final_v2 (มี augmentation) ทั้งคู่ใช้ fold seed 42
o1 = np.load("t1/research/v1_best_oof.npy"); o2 = np.load("t1/research/best_oof.npy")
nm = lab[SIGNAL].isna().sum(1).clip(upper=4).values
t = pd.DataFrame({k: {"rows": int((nm == k).sum()), "AUC no-aug": roc_auc_score(y[nm == k], o1[nm == k]), "AUC masking-aug": roc_auc_score(y[nm == k], o2[nm == k])} for k in range(5)}).T
t["gain"] = t["AUC masking-aug"] - t["AUC no-aug"]; t.index.name = "จำนวนค่าที่หาย (4 = 4+)"
print("รวม: no-aug %.5f → aug %.5f" % (roc_auc_score(y, o1), roc_auc_score(y, o2)))
t.round(5)
''')]

C += [md(r'''
**ผล (3 seeds):** 2 สำเนา +0.0006 AUC, **4 สำเนา +0.0007 AUC บนทุก seed** (sd 0.00002) · ซ่อน 2 เท่าของอัตราจริงไม่ดีขึ้น
สูตรสุดท้าย **`final_v2`** = fe_v2 + masking ×4 + LightGBM lr 0.02 · 3 bags × 5 folds × 3 seeds → **AUC 0.8948, binary F2 nested 0.8391**

## 10. Metric จริงบน leaderboard = binary F2 และการเลือกจุดตัด
โมเดลให้ความน่าจะเป็น ต้องเลือกว่า **ทายคนโกงกี่เปอร์เซ็นต์** จุดตัดที่ดีที่สุดขึ้นกับ metric มาก
'''), code(r'''
p = np.load("t1/research/best_oof_avg3.npy")  # OOF เฉลี่ย 3 split seeds (= โมเดลที่ใช้ทาย test)
order = np.argsort(-p); ys = y[order]; tpc = np.cumsum(ys); P = ys.sum(); n = len(y)
rates = np.linspace(.25, .85, 241); fb, fm = [], []
for r_ in rates:
    k = int(r_ * n); tp = tpc[k - 1]; fp = k - tp; fn = P - tp; tn = n - k - fn
    f_pos = 5 * tp / (5 * tp + 4 * fn + fp); f_neg = 5 * tn / (5 * tn + 4 * fp + fn)
    fb.append(f_pos); fm.append((f_pos + f_neg) / 2)
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(rates, fb, label="binary F2 (LB metric)"); ax.plot(rates, fm, label="macro F2 (first assumption)")
for r_, c in [(rates[np.argmax(fm)], "tab:orange"), (rates[np.argmax(fb)], "tab:blue"), (.60, "k")]: ax.axvline(r_, color=c, ls="--", lw=1)
ax.set_xlabel("fraction predicted cheater"); ax.legend(); ax.set_title("Score vs cut (OOF)"); plt.show()
print("macro F2 best at %.1f%% positive, binary F2 best at %.1f%% positive" % (100 * rates[np.argmax(fm)], 100 * rates[np.argmax(fb)]))
''')]

C += [md(r'''
**การพิสูจน์บน leaderboard (probe):** ส่งไฟล์สองไฟล์ที่ **ต่างกันแค่จุดตัดของ T1**

| ไฟล์ | T1 ทายโกง | Public LB |
|---|---|---|
| sub_v10d | ~41% (จุดตัด macro) | 0.66614 |
| probe_v10e | ~63% (จุดตัด binary F2) | **0.67890 (+0.0128)** |

คะแนน LB ≈ 0.2 × (T1+T2+T3+T4+T5) → T1 บน LB ขึ้น ≈ 0.064 ตรงกับที่ binary F2 ทำนาย และขัดกับ macro F2 (ซึ่งจะต้องลดลง)
→ **ยืนยันว่า LB ใช้ binary F2** — ขั้นตอนนี้ขั้นเดียวได้มากกว่าการจูนโมเดลทั้งหมดรวมกัน

ปรับจุดตัดละเอียดบน LB: 60% = ดีที่สุด, 59.2% −0.00003, 61% −0.00002 → กราฟแบนมาก ใช้ **60%**

## 11. Rank blend สุดท้าย
รวมโมเดลสามแหล่งที่สร้างต่างกัน โดยใช้ **อันดับ** (rank) แทนความน่าจะเป็น เพราะแต่ละโมเดลมีสเกลต่างกัน
- 0.7 × `final_v2` (LightGBM + masking)
- 0.2 × stacking logistic regression (stack ของ ET, kNN, NB, MLP, SVM ฯลฯ จากการทดลองอีกสาย)
- 0.1 × GBM pipeline (XGB + LGB + CatBoost, Optuna 200 trials)
'''), code(r'''
R = lambda v: rankdata(v) / len(v)
s_lr = np.load("t1/research/blind_sonnet/stack_lr_oof.npy"); gbm = np.load("t1/preds/t1_pipeline/oof_blend.npy")
cands = {"final_v2 alone": R(p), "0.8/0.1/0.1": .8 * R(p) + .1 * R(s_lr) + .1 * R(gbm), "0.7/0.2/0.1 (used)": .7 * R(p) + .2 * R(s_lr) + .1 * R(gbm),
         "stack_lr alone": R(s_lr), "GBM pipeline alone": R(gbm)}
tab = {}
for k, v in cands.items():
    sc = [nested_thr_score(y, v, folds_for(y, s), "binary") for s in range(10)]
    tab[k] = {"AUC": roc_auc_score(y, v), "binary F2 nested (mean of 10 fold splits)": np.mean(sc)}
pd.DataFrame(tab).T.round(5)
''')]

C += [md(r'''
บน LB: v20a (rank blend) **+0.00017** เทียบกับ v18a, ใกล้ค่าที่ทำนายไว้ +0.00014 · v20b (0.8/0.1/0.1) +0.00013

## 12. สิ่งที่ลองแล้วไม่ได้ผล + เพดานของข้อมูล
| ไอเดีย | ผล | เหตุผล |
|---|---|---|
| Pseudo-labeling บน 13,184 แถวไม่มี label (exp07, exp11, exp19, round13) | −0.0001 ถึง −0.0006 CV, LB −0.00006/−0.00008 | แม้ให้ label **จริง** ทุกแถวยังได้แค่ +0.001 (upper bound); teacher ไม่มีข้อมูลใหม่ให้ student |
| Optuna re-tune (LGB 120, XGB 50, Cat 40 trials) | ได้บน split ที่จูน ไม่ transfer | พารามิเตอร์ baseline อยู่ที่ optimum แล้ว |
| Class weight / focal loss / early stopping บน partial AUC | < 0.001 | จุดตัดจัดการ F2 อยู่แล้ว |
| ลบแถวที่ "น่าจะ label ผิด" (2–10%) | แย่ลง | แถวยากคือการซ้อนทับจริง ไม่ใช่ label ผิด (label ผิดมีแค่ ~0.12%) |
| kNN target rate, imputer, ratio features, MLP, Mixture NB | ไม่ชนะ | GBM จับได้หมดแล้ว |
| เพิ่ม seed เป็น 13 seed | +0.00008 = noise | ความแปรปรวนต่ำพออยู่แล้ว |
| Per-group threshold / calibration | ไม่ดีขึ้น | จุดตัดเดียวบน posterior เหมาะที่สุดสำหรับ F-measure |

**เพดาน (exp18):** learning curve (10% → 100% ของข้อมูล) fit power law ได้ asymptote AUC ≈ 0.8939 · ค้นหาสูตรซ่อนอื่นด้วยการทาย 31 คอลัมน์จากคอลัมน์อื่น → พบแค่สูตร lps อันเดียว
→ โมเดลสุดท้าย (AUC 0.8948) **อยู่ที่เพดานข้อมูลแล้ว**

## 13. สรุปไฟล์ส่ง
'''), code(r'''
o = pd.read_csv("t1/research/t1_best_test.csv"); pp = pd.read_csv("t1/preds/t1_pipeline/t1_blend.csv"); st = np.load("t1/research/blind_sonnet/stack_lr_test.npy")
b = .7 * R(o.score.values) + .2 * R(st) + .1 * R(pp.score.values)
pred = (b >= np.quantile(b, 1 - .60)).astype(float)
sub = pd.read_csv("submissions/FINAL/sub_v24b_noleak_T3seedbag10_T4blendEva02.csv")
print("ทายโกง %.2f%% ของ test | ตรงกับไฟล์ส่งจริง (sub_v24b):" % (100 * pred.mean()), bool((pred == sub.task1.values).all()))
'''), md(r'''
| ขั้น | สิ่งที่ได้ |
|---|---|
| EDA → ตัด noise 16 คอลัมน์ | AUC +0.0004 |
| ค้นพบสูตร `lps = level·r/age` → zone posterior (fe_v2) | AUC +0.0007 |
| lr 0.02 + bagging | AUC +0.0006 |
| MCAR masking augmentation | AUC +0.0004–0.0007 |
| **พิสูจน์ metric = binary F2 → ทายโกง 60%** | **LB +0.0128** |
| Rank blend 0.7/0.2/0.1 | LB +0.00017 |

**บทเรียนหลัก**
1. หาเพดานของข้อมูลก่อน จะได้รู้ว่าควรลงแรงตรงไหน
2. ข้อมูลสังเคราะห์มักมีสูตรซ่อน — การหาสูตรได้ผลมากกว่าการจูนโมเดล
3. **ต้องรู้ metric จริง** — probe บน LB แบบเปลี่ยนทีละอย่าง
4. ทุกการตัดสินใจใช้ nested CV + หลาย seed และเชื่อเฉพาะผลที่เกิน noise
''')]

build("T1_cheater_detection.ipynb", C)
