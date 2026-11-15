import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbtools import build, md, code, SETUP

C = []
C += [md(r'''
# Task 2 — Player Segmentation (4-class Classification)
**ทีม ML003 · Player Intelligence System (CPE342)**

เป้าหมาย: ทาย `segment` (0–3) ของผู้เล่นจากพฤติกรรมการเล่น การใช้จ่าย และสังคม · **metric = macro F1**
Notebook นี้แสดงที่มาของทุกการตัดสินใจ: ทำไมเลือก **Naive Bayes** (ไม่ใช่ GBM), การหา density ของแต่ละ feature, การจัดการ label noise 3% ด้วย EM, การผสมโมเดล และกฎตัดสินใจที่เหมาะกับ macro F1
ไฟล์ส่งแบบไม่ใช้ leak (`sub_v24b`) ใช้เฉพาะ feature — ส่วน player_id leak แสดงไว้ท้ายเพื่อความครบถ้วนเท่านั้น

| หัวข้อ | เนื้อหา |
|---|---|
| 1 | ข้อมูลและ metric |
| 2 | Feature ไหนมีสัญญาณ (ตัด noise 16 คอลัมน์) |
| 3 | ค่าที่หายเป็น MCAR |
| 4 | **Feature เป็นอิสระต่อกันเมื่อรู้คลาส → Naive Bayes คือ Bayes-optimal** |
| 5 | โครงสร้างแฝง 2×2 (engagement × spending) |
| 6 | Histogram Naive Bayes จากศูนย์ |
| 7 | เลือกความละเอียดของ density ด้วย log-loss (ไม่ใช่ F1) |
| 8 | Label noise 3% → EM Naive Bayes |
| 9 | เทียบกับโมเดลอื่น + เพดานของข้อมูล |
| 10 | Blend + กฎตัดสินใจ affine สำหรับ macro F1 |
| 11 | (ภาคผนวก) player_id leak |
| 12 | สรุป |
'''), code(SETUP + r'''
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score, log_loss
tr = pd.read_csv("cpe342-kaggle-main/02_multiclass_classification/train.csv")
te = pd.read_csv("cpe342-kaggle-main/02_multiclass_classification/test.csv")
y = tr.segment.astype(int).values; K = 4; N = len(tr)
print("train", tr.shape, "| test", te.shape)
print("class share:", np.round(np.bincount(y) / N, 4))
f1m = lambda yt, P: f1_score(yt, np.asarray(P).argmax(1), average="macro")
''')]

C += [md(r'''
## 1. ข้อมูลและ metric
- 104,000 แถว, 44 feature (ตัวเลข + categorical เช่น region, platform, loyalty_tier, engagement_level)
- 4 คลาส สัดส่วน ~40 / 25 / 20 / 15%
- **macro F1** = เฉลี่ย F1 ของ 4 คลาสเท่า ๆ กัน → คลาสเล็กสำคัญเท่าคลาสใหญ่ ทำให้ "กฎตัดสินใจ" มีผล (ข้อ 10)
- Validation: 5-fold StratifiedKFold, seeds 42 / 7 / 123
'''), code(r'''
feats = [c for c in tr.columns if c not in ("player_id", "segment")]
is_disc = {c: (not pd.api.types.is_numeric_dtype(tr[c])) or tr[c].nunique() <= 12 for c in feats}
print("continuous:", sum(not v for v in is_disc.values()), "| discrete/categorical:", sum(is_disc.values()))
tr[feats].head(3).T
''')]

C += [md(r'''
## 2. Feature ไหนมีสัญญาณ
วัด **information gain ของแต่ละ feature เดี่ยว**: ทำ Naive Bayes ที่มี feature เดียว (5-fold) แล้วดูว่า log-loss ลดลงจากการทายด้วย prior อย่างเดียวเท่าไร
ถ้า ≤ 0 = feature นั้นไม่ช่วยทายคลาสเลย
'''), code(r'''
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))

def codes(c, a, nbins):
    """bin index for every train row + test row, edges/levels fitted on train rows a only. returns (bt, be, n, nan_t, nan_e)"""
    if is_disc[c]:
        vt = tr[c].astype(object).where(tr[c].notna(), "NA").astype(str).values
        ve = te[c].astype(object).where(te[c].notna(), "NA").astype(str).values
        lv = {x: i for i, x in enumerate(sorted(set(vt[a])))}
        bt = np.array([lv.get(x, -1) for x in vt]); be = np.array([lv.get(x, -1) for x in ve])
        return np.maximum(bt, 0), np.maximum(be, 0), len(lv), bt < 0, be < 0      # NaN = its own level
    x, xe = tr[c].values.astype(float), te[c].values.astype(float)
    e = np.unique(np.nanquantile(x[a], np.linspace(0, 1, nbins + 1))); n = len(e) - 1
    bt = np.clip(np.searchsorted(e, np.nan_to_num(x), side="right") - 1, 0, n - 1)
    be = np.clip(np.searchsorted(e, np.nan_to_num(xe), side="right") - 1, 0, n - 1)
    return bt, be, n, np.isnan(x), np.isnan(xe)                                    # NaN = marginalised (skipped)

def class_logdens(c, b, n, W, sm, alpha=1.0):
    """log p(bin | class) from (soft) class weights W (rows x K) on training rows."""
    out = np.zeros((n, K))
    for k in range(K):
        cnt = np.bincount(b, weights=W[:, k], minlength=n)
        if sm > 0 and not is_disc[c]: cnt = gaussian_filter1d(cnt, sm, mode="nearest")
        out[:, k] = np.log((cnt + alpha) / (cnt + alpha).sum())
    return out

prior_ll = log_loss(y, np.tile(np.bincount(y) / N, (N, 1)))
gain = {}
for c in feats:
    P = np.zeros((N, K))
    for a, b in FOLDS:
        bt, _, n, nt, _ = codes(c, a, 32); keep = a[~nt[a]] if not is_disc[c] else a
        L = class_logdens(c, bt[keep], n, np.eye(K)[y[keep]], 0.7)[bt[b]] + np.log(np.bincount(y[a]) / len(a))
        if not is_disc[c]: L[nt[b]] = np.log(np.bincount(y[a]) / len(a))
        P[b] = np.exp(L - L.max(1, keepdims=True)); P[b] /= P[b].sum(1, keepdims=True)
    gain[c] = prior_ll - log_loss(y, P)
g = pd.Series(gain).sort_values(ascending=False)
NOISE = list(g[g <= 0.0005].index)
ax = g.plot.bar(figsize=(13, 3.5), color=["tab:gray" if c in NOISE else "tab:blue" for c in g.index]); ax.set_title("univariate information gain (log-loss drop vs prior); gray = noise"); plt.tight_layout(); plt.show()
print("noise columns (%d):" % len(NOISE), NOISE)
''')]

C += [md(r'''
**ผล:** 16 คอลัมน์ไม่มีข้อมูลเกี่ยวกับคลาสเลย (รวม `random_metric_1/2/3`, region, platform, device, payment, language, account_age_days ฯลฯ) → **ตัดทิ้ง** เหลือ 28 feature
(ใน NB การเก็บ feature noise ไว้ไม่ได้ช่วย แต่จะเพิ่ม variance ของ density ที่ประมาณ)

## 3. ค่าที่หายเป็น MCAR
'''), code(r'''
FE = [c for c in feats if c not in NOISE]
na = pd.DataFrame({k: tr.loc[y == k, FE].isna().mean() for k in range(K)}); na.columns = [f"class {k}" for k in range(K)]
print("NaN rate ต่อคลาส (ช่วงของทุก feature): min %.4f  max %.4f" % (na.values.min(), na.values.max()))
print("จำนวน NaN ต่อแถว เฉลี่ยต่อคลาส:", np.round([tr.loc[y == k, FE].isna().sum(1).mean() for k in range(K)], 3))
na.head(8).round(4)
'''), md(r'''
อัตราการหาย ~7.5–8% เท่ากันทุกคลาส → การหายไม่บอกคลาส → ใน Naive Bayes **ข้าม (marginalise) feature ที่หาย** ได้อย่างถูกต้องตามทฤษฎี: p(x ที่เหลือ | class) = ผลคูณของ feature ที่ไม่หายเท่านั้น
(feature แบบ discrete ให้ NaN เป็นระดับของตัวเอง)

## 4. Feature เป็นอิสระต่อกันเมื่อรู้คลาส → Naive Bayes คือโมเดลที่ดีที่สุด
Naive Bayes สมมติว่า p(x | class) = Π p(x_j | class) ถ้าข้อมูลถูกสร้างแบบนี้จริง NB คือ **Bayes-optimal classifier** (ไม่มีโมเดลไหนดีกว่าได้)
ทดสอบ: correlation ระหว่าง feature **ทั้งหมดรวม** vs **ภายในแต่ละคลาส**
'''), code(r'''
CONT = [c for c in FE if not is_disc[c]]
corr_all = tr[CONT].corr(method="spearman").values
corr_in = np.mean([tr.loc[y == k, CONT].corr(method="spearman").values for k in range(K)], 0)
off = ~np.eye(len(CONT), dtype=bool)
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, M, t in [(axs[0], corr_all, "all rows"), (axs[1], corr_in, "within class (avg of 4)")]:
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r"); ax.set_title(f"Spearman correlation, {t}"); ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=axs, shrink=.8); plt.show()
print("|corr| เฉลี่ยนอกแนวทแยง: รวมทุกแถว %.3f → ภายในคลาส %.3f (max %.3f)" % (np.abs(corr_all[off]).mean(), np.abs(corr_in[off]).mean(), np.abs(corr_in[off]).max()))
''')]

C += [md(r'''
**ผล:** feature สัมพันธ์กันเมื่อดูรวม (เพราะคลาสต่างกันมีค่าเฉลี่ยต่างกัน) แต่ **เกือบเป็นศูนย์ภายในแต่ละคลาส**
→ ข้อมูลถูกสร้างแบบ "เลือกคลาสก่อน แล้วสุ่มแต่ละ feature อิสระ" → Naive Bayes ถูกต้องตามโครงสร้าง และโมเดลซับซ้อน (GBM) ไม่มี interaction ให้หาเพิ่ม

## 5. โครงสร้างแฝง 2×2
ค่าเฉลี่ยของแต่ละคลาสบนแกน engagement (เวลาเล่น) กับ spending (การใช้จ่าย):
'''), code(r'''
cm = tr.groupby("segment")[["total_playtime_hours", "play_frequency", "total_spending_thb", "avg_monthly_spending"]].median()
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.scatter(cm.total_playtime_hours, cm.total_spending_thb, s=300)
for k, r_ in cm.iterrows(): ax.annotate(f"seg {k}", (r_.total_playtime_hours, r_.total_spending_thb), ha="center", va="center", color="w", fontsize=9)
ax.set_xlabel("median total_playtime_hours (engagement)"); ax.set_ylabel("median total_spending_thb (spending)"); ax.set_title("class medians"); plt.show()
cm.round(1)
'''), md(r'''
คลาสแบ่งตามสองแกน (เล่นมาก/น้อย × จ่ายมาก/น้อย) — ช่วยอธิบายว่า feature กลุ่มไหนแยกคลาสคู่ไหน
เราทดลอง "2×2-factorised NB" (บังคับให้ density ใช้ร่วมกันตามแกน) แต่ข้อมูลปฏิเสธการรวม 26/28 feature → ใช้ NB ปกติที่ density แยกต่อคลาส

## 6. Histogram Naive Bayes จากศูนย์
สำหรับแต่ละ feature ต่อเนื่อง:
1. แบ่ง bin ตาม **quantile** ของแถวเทรน (bin ละเท่า ๆ กัน)
2. นับจำนวนแถวของแต่ละคลาสในแต่ละ bin → **Gaussian smoothing** σ (หน่วย bin) ให้ density เรียบ
3. Laplace α = 1 กันค่าศูนย์ → log p(bin | class)

ทายคลาส: log π_k + Σ_j log p(x_j | k) → softmax → argmax
'''), code(r'''
def nb_fit_predict(a, b, cfg, W=None, test=False):
    """returns log-joint (rows x K) for rows b (or test) from training rows a; W = soft class weights for EM (default one-hot)."""
    W = np.eye(K)[y[a]] if W is None else W
    pi = W.sum(0) / W.sum(); L = np.tile(np.log(pi), (len(te) if test else len(b), 1))
    for c in FE:
        nb_, sm = cfg.get(c, (32, 0.7))
        bt, be, n, nt, ne = codes(c, a, nb_)
        keep = ~nt[a] if not is_disc[c] else np.ones(len(a), bool)
        ld = class_logdens(c, bt[a][keep], n, W[keep], sm)
        if test: l = ld[be]; l[ne & (not is_disc[c])] = 0
        else: l = ld[bt[b]]; l[nt[b] & (not is_disc[c])] = 0
        L += l
    return L

def softmax(L): P = np.exp(L - L.max(1, keepdims=True)); return P / P.sum(1, keepdims=True)

def cv_nb(cfg, folds=FOLDS):
    P = np.zeros((N, K))
    for a, b in folds: P[b] = softmax(nb_fit_predict(a, b, cfg))
    return P

P_uni = cv_nb({})          # 32 bins, sigma 0.7 for every feature
print("NB uniform (32 bins, σ 0.7): macro F1 %.4f | log-loss %.4f" % (f1m(y, P_uni), log_loss(y, P_uni)))
''')]

C += [md(r'''
## 7. เลือกความละเอียดของ density ด้วย log-loss
baseline เดิมใช้ Optuna จูน (bins, σ) ต่อ feature โดยตรงบน macro F1 → ได้ 0.7557 แต่เมื่อทำ **nested** (จูนใน fold ด้านใน) เหลือ **0.7544** = overfit (จูน 46 พารามิเตอร์บนแถวชุดเดียวกัน)

วิธีที่ถูก: เลือก (bins, σ) ของแต่ละ feature จาก **log-loss ของ posterior แบบ univariate** ด้วย 4-fold CV ภายใน
- log-loss วัดคุณภาพของ density ทั้งเส้น (เสถียรกว่า F1 ที่ขึ้นกับจุดตัด)
- grid: bins {64, 128, 256} × σ {3, 4, 6, 8, 12} → เลือกได้ density ที่ **ละเอียดและเรียบกว่า** default มาก
'''), code(r'''
cfg = {c: tuple(v) for c, v in json.load(open("t2/research/best_cfg.json")).items()}
print(pd.Series({c: cfg[c] for c in CONT if c in cfg}).to_string())
P_sel = cv_nb(cfg)
print("\nNB uniform        : macro F1 %.4f | log-loss %.4f" % (f1m(y, P_uni), log_loss(y, P_uni)))
print("NB LL-selected cfg: macro F1 %.4f | log-loss %.4f" % (f1m(y, P_sel), log_loss(y, P_sel)))
''')]

C += [md(r'''
## 8. Label noise 3% → EM Naive Bayes
จากการวิเคราะห์ (ภาคผนวก ข้อ 11) พบว่า **~3% ของ label ถูกสลับแบบสุ่ม** ไปเป็นคลาสอื่นอย่างสม่ำเสมอ
Model: มีคลาสจริง z ที่ซ่อนอยู่ และ label ที่เห็น ỹ ผิดด้วยความน่าจะเป็น ε (กระจายเท่ากันไปอีก 3 คลาส)

- **E-step:** r_iz ∝ π_z · Π_j p_j(x_ij | z) · P(ỹ_i | z)
- **M-step:** ประมาณ density ใหม่จากน้ำหนัก r (soft counts), π และ ε = 1 − mean(r ของ label ที่เห็น)
- **ทำนาย:** P(ỹ | x) = softmax(log π + Σ log p_j) · M_ε

EM ประมาณ ε ได้ ≈ 0.028 จาก label อย่างเดียว (ค่าจริงจาก ID analysis = 0.030)
ผลคือ argmax เกือบเหมือนเดิม แต่ **ความน่าจะเป็นถูก calibrate ดีขึ้นมาก** (log-loss 0.623 → 0.599) ซึ่งสำคัญตอน blend และตอนใช้กฎ affine
'''), code(r'''
def em_nb_cv(cfg, iters=15, folds=FOLDS):
    P = np.zeros((N, K)); eps_list = []
    for a, b in folds:
        Y = np.eye(K)[y[a]]; R = Y.copy(); eps = 0.03
        for it in range(iters):
            M = np.full((K, K), eps / 3); np.fill_diagonal(M, 1 - eps)
            if it: R = softmax(nb_fit_predict(a, a, cfg, W=R) + np.log(Y @ M.T)); eps = 1 - (R * Y).sum() / len(a)
        M = np.full((K, K), eps / 3); np.fill_diagonal(M, 1 - eps)
        P[b] = softmax(nb_fit_predict(a, b, cfg, W=R)) @ M; eps_list.append(eps)
    return P, eps_list
P_em, eps = em_nb_cv(cfg, iters=8)
print("ε ที่ EM ประมาณได้ต่อ fold:", np.round(eps, 4))
print("EM-NB (LL-selected cfg): macro F1 %.4f | log-loss %.4f" % (f1m(y, P_em), log_loss(y, P_em)))
P_best = np.load("t2/research/best_oof_proba.npy")
print("ไฟล์ OOF สุดท้ายที่บันทึกไว้ (15 iterations, nested cfg): macro F1 %.4f | log-loss %.4f" % (f1m(y, P_best), log_loss(y, P_best)))
''')]

C += [md(r'''
## 9. เทียบกับโมเดลอื่น + เพดานของข้อมูล
(ผลจริง 3 seeds จาก `t2/research/REPORT.md`)

| โมเดล | macro F1 | log-loss | หมายเหตุ |
|---|---|---|---|
| NB uniform | 0.7550 | – | baseline |
| NB Optuna ต่อ feature (nested) | 0.7544 | 0.624 | overfit |
| **EM-NB + LL-selected density** | **0.7562 ± 0.0002** | **0.599** | **ใช้จริง** |
| LightGBM raw / จูน nested | 0.7509 / 0.7524 | 0.617 | overfit ~0.04 (train F1 0.79) |
| Spline GAM (multinomial) | 0.7529 | 0.620 | LL ดีกว่าแต่ F1 แย่กว่า |
| WANBIA (ถ่วงน้ำหนัก feature) | 0.7541 | 0.620 | – |
| Class specialists (one-vs-rest LGBM) | 0.7537–0.7551 | – | AUC เท่ากับ NB ทุกคลาส |
| 2×2 factorised NB | 0.7550 | – | ข้อมูลไม่รองรับการรวม density |

**Learning curve:** 10% → 100% ของข้อมูลได้ 0.746 → 0.7557 เพิ่ม ~0.0018 ต่อการเพิ่มข้อมูลเท่าตัวและลดลงเรื่อย ๆ → เหลือ headroom แค่ ~0.002 → **0.757 ≈ เพดานแบบใช้ feature อย่างเดียว**

## 10. Blend + กฎตัดสินใจ affine สำหรับ macro F1
**Blend:** 0.87 × EM-NB + 0.08 × spline logistic regression + 0.05 × MLP (โมเดลจากการทดลองอีกสาย) — ให้ความน่าจะเป็นที่หลากหลายขึ้นเล็กน้อย

**ทำไม argmax ไม่พอ:** argmax เหมาะกับ accuracy แต่ macro F1 ให้รางวัลคลาสเล็กมากกว่า
ทฤษฎี: กฎที่ทำ F1 สูงสุดมีรูป **affine ในความน่าจะเป็น** — ทาย k ที่ทำให้ a_k·p_k + b_k มากสุด โดย
a_k = 2 / (π_k + q_k), b_k = −F_k / (π_k + q_k) (π = สัดส่วนจริง, q = สัดส่วนที่ทาย, F_k = F1 ของคลาส k)
เนื่องจาก q และ F ขึ้นกับคำตอบเอง จึงหาแบบ **fixed-point iteration** (วนจนนิ่ง ~8 รอบ)
'''), code(r'''
S = "t2/research/blind_sonnet/"
Bo = .87 * P_best + .08 * np.load(S + "logreg_spline_A_oof.npy") + .05 * np.load(S + "mlp_torch_A_oof.npy")
def affine_rule(P, ytrue, iters=8):
    pi = np.bincount(ytrue, minlength=K) / len(ytrue); q = pi.copy(); Fk = np.full(K, .75)
    for _ in range(iters):
        a = 2 / (pi + q); b = -Fk / (pi + q); pr = (a * P + b).argmax(1); q = np.bincount(pr, minlength=K) / len(ytrue)
        Fk = np.array([2 * ((pr == k) & (ytrue == k)).mean() / (pi[k] + q[k]) for k in range(K)])
    return a, b
a_, b_ = affine_rule(Bo, y)
print("EM-NB argmax          : %.5f" % f1m(y, P_best))
print("blend argmax          : %.5f" % f1m(y, Bo))
print("blend + affine rule   : %.5f" % f1_score(y, (a_ * Bo + b_).argmax(1), average="macro"))
print("rows changed by the rule:", int(((a_ * Bo + b_).argmax(1) != Bo.argmax(1)).sum()), "| a =", np.round(a_, 3), "b =", np.round(b_, 3))
'''), code(r'''
# nested check: fit the rule on 4 folds, apply on the 5th (rule has 8 numbers, so nesting matters)
pr_n = np.zeros(N, int)
for a, b in FOLDS:
    aa, bb = affine_rule(Bo[a], y[a]); pr_n[b] = (aa * Bo[b] + bb).argmax(1)
print("blend + affine (nested): %.5f  vs EM-NB argmax %.5f" % (f1_score(y, pr_n, average="macro"), f1m(y, P_best)))
''')]

C += [md(r'''
บน LB: sub_v18a (blend + affine, 533 แถวใน test เปลี่ยน) = **+0.00034** เทียบกับ v16b → T2 บน LB ≈ +0.0017 (CV ทำนาย +0.0009)

## 11. (ภาคผนวก) player_id leak — **ไม่ใช้ในไฟล์ no-leak**
ระหว่างวิเคราะห์ label noise พบว่า `player_id` ถูกแจกเป็นช่วง ๆ ตาม segment:
'''), code(r'''
pid = tr.player_id.str[1:].astype(int)
fig, ax = plt.subplots(figsize=(11, 2.8))
ax.scatter(pid, y + np.random.default_rng(0).uniform(-.3, .3, N), s=.2, alpha=.3); ax.set_xlabel("player_id number"); ax.set_ylabel("segment"); ax.set_title("segment vs player_id")
for cut in (52000, 84500, 110500): ax.axvline(cut, color="r", lw=1)
plt.show()
blk = np.select([pid < 52000, pid < 84500, pid < 110500], [0, 1, 2], 3)
print("label ≠ block: %.2f%% (กระจายเท่า ๆ กันไปคลาสอื่น = label ที่ถูกสลับแบบสุ่ม)" % (100 * (blk != y).mean()))
print("macro F1 ถ้าใช้ block rule: %.4f (เพดาน เพราะ 3%% สลับแบบสุ่ม)" % f1_score(y, blk, average="macro"))
''')]

C += [md(r'''
- ID ถูกแจกเป็นบล็อก 40 / 25 / 20 / 15% ของ 130,000 → ใช้ทายได้ macro F1 0.968
- 3% ที่ไม่ตรงบล็อกคือ **label ที่สลับแบบสุ่ม** (feature ตรงกับคลาสของบล็อก 78.6% เท่ากับความแม่นยำปกติของ NB) → นี่คือที่มาของ ε ≈ 0.03 ในข้อ 8
- เป็น artefact ของการสร้างข้อมูล ไม่ใช่ความรู้จาก feature → **ไฟล์ no-leak ไม่ใช้**; ไฟล์ leak ใช้ (รอ TA ยืนยัน)

## 12. สรุป
'''), code(r'''
Bt = .87 * np.load("t2/research/best_test_proba.npy") + .08 * np.load(S + "logreg_spline_A_test.npy") + .05 * np.load(S + "mlp_torch_A_test.npy")
t2 = (a_ * Bt + b_).argmax(1)
sub = pd.read_csv("submissions/FINAL/sub_v24b_noleak_T3seedbag10_T4blendEva02.csv")
print("test class share:", np.round(np.bincount(t2, minlength=K) / len(t2), 4))
print("ตรงกับไฟล์ส่ง sub_v24b:", bool((t2 == sub.task2.values.astype(int)).all()))
'''), md(r'''
| ขั้น | ผล (macro F1 CV) |
|---|---|
| EDA: feature อิสระต่อกันในคลาส → เลือก Naive Bayes | NB ชนะ GBM 0.004 |
| ตัด noise 16 คอลัมน์ + marginalise ค่าที่หาย (MCAR) | – |
| เลือก density ด้วย log-loss แทนการจูนบน F1 | 0.7544 (nested Optuna) → 0.7558 |
| EM สำหรับ label noise 3% | 0.7562, log-loss 0.599 |
| Blend + affine F1 rule | nested +0.0009, LB +0.00034 |

**บทเรียนหลัก**
1. ดูโครงสร้างการสร้างข้อมูลก่อนเลือกโมเดล — conditional independence ⇒ NB เป็นคำตอบที่ถูก ไม่ใช่โมเดลใหญ่
2. จูนพารามิเตอร์จำนวนมากบน F1 = overfit; ใช้เกณฑ์ที่เสถียร (log-loss) และวัดแบบ nested
3. Macro F1 ต้องการกฎตัดสินใจของตัวเอง ไม่ใช่ argmax
''')]

build("T2_player_segmentation.ipynb", C)
