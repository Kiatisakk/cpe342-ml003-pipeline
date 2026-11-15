"""
Baseline Task 1–3 ด้วย HistGradientBoosting (scikit-learn, ไม่ต้องลงเพิ่ม) + 5-fold CV ด้วย metric จริง
ใช้:  python t123_baseline.py t1 [fold ...]   → เทรนเฉพาะ fold ที่ระบุ (ไม่ระบุ = ทุก fold)
      python t123_baseline.py t1 report        → รวม OOF, หา threshold/ตัวเลือกที่ดีที่สุด, เขียน preds
test prediction = เฉลี่ยโมเดลของทุก fold · ไม่ submit อะไร
"""
import os, sys, time, glob, json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from sklearn.metrics import fbeta_score, f1_score
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "cpe342-kaggle-main")
WORK = os.path.join(HERE, "work"); os.makedirs(WORK, exist_ok=True)
CFG = {"t1": ("01_binary_classification", "is_cheater"),
       "t2": ("02_multiclass_classification", "segment"),
       "t3": ("03_regression", "spending_30d")}
SEED, NF = 42, 5


def load(task):
    folder, tgt = CFG[task]
    tr = pd.read_csv(f"{DATA}/{folder}/train.csv"); te = pd.read_csv(f"{DATA}/{folder}/test.csv")
    tr = tr[tr[tgt].notna()].reset_index(drop=True)                     # Task 1 มี target ว่าง ~12.7%
    y = tr[tgt].values
    feats = [c for c in te.columns if c != "player_id"]
    cats = [c for c in feats if tr[c].dtype == object]
    Xtr, Xte = tr[feats].copy(), te[feats].copy()
    if cats:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
        Xtr[cats] = enc.fit_transform(Xtr[cats].astype("object")); Xte[cats] = enc.transform(Xte[cats].astype("object"))
    catmask = np.array([c in cats for c in feats])
    return Xtr.values.astype(np.float32), y, Xte.values.astype(np.float32), catmask


def model(task, catmask, kind="main"):
    common = dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=SEED,
                  categorical_features=catmask if catmask.any() else None)
    if task == "t1": return HGBC(**common)
    if task == "t2": return HGBC(class_weight="balanced", **common)
    if kind == "clf": return HGBC(**common)                              # t3 stage 1: จะจ่ายไหม
    return HGBR(loss="absolute_error", **common)                          # t3: MAE ↔ NMAE


def folds(task, y):
    strat = (y > 0).astype(int) if task == "t3" else y.astype(int)
    return list(StratifiedKFold(NF, shuffle=True, random_state=SEED).split(np.zeros(len(y)), strat))


def fit_fold(task, k):
    Xtr, y, Xte, cm = load(task); tr, va = folds(task, y)[k]; t0 = time.time(); out = {}
    if task in ("t1", "t2"):
        m = model(task, cm).fit(Xtr[tr], y[tr].astype(int))
        out["va"] = m.predict_proba(Xtr[va]); out["te"] = m.predict_proba(Xte)
    else:
        m = model(task, cm).fit(Xtr[tr], y[tr])                           # (a) MAE regressor ทุกแถว
        out["va_a"] = m.predict(Xtr[va]); out["te_a"] = m.predict(Xte)
        c = model(task, cm, "clf").fit(Xtr[tr], (y[tr] > 0).astype(int)) # (b) two-stage
        pos = tr[y[tr] > 0]; r = model(task, cm).fit(Xtr[pos], y[pos])
        out["va_p"] = c.predict_proba(Xtr[va])[:, 1]; out["te_p"] = c.predict_proba(Xte)[:, 1]
        out["va_r"] = r.predict(Xtr[va]); out["te_r"] = r.predict(Xte)
    np.savez(f"{WORK}/{task}_fold{k}.npz", va_idx=va, **out)
    print(f"{task} fold {k}: {time.time()-t0:.0f}s")


def f2_macro(y, p): return fbeta_score(y, p, beta=2, average="macro")
def nmae(y, p): return 1 / (1 + np.abs(y - p).sum() / y.sum())


def report(task):
    Xtr, y, Xte, _ = load(task); fs = sorted(glob.glob(f"{WORK}/{task}_fold*.npz")); assert len(fs) == NF, fs
    R = [np.load(f) for f in fs]; res = {}
    if task in ("t1", "t2"):
        oof = np.zeros((len(y), R[0]["va"].shape[1])); te = np.mean([r["te"] for r in R], 0)
        for r in R: oof[r["va_idx"]] = r["va"]
        yi = y.astype(int)
        if task == "t1":
            ths = np.arange(0.05, 0.95, 0.01)
            sc = [f2_macro(yi, (oof[:, 1] >= t).astype(int)) for t in ths]; best = ths[int(np.argmax(sc))]
            res = {"F2_macro@0.5": f2_macro(yi, (oof[:, 1] >= .5).astype(int)), "best_thr": float(best),
                   "F2_macro@best": max(sc), "F2_class1@best": fbeta_score(yi, (oof[:, 1] >= best).astype(int), beta=2)}
            pred = (te[:, 1] >= best).astype(int)
        else:
            res = {"F1_macro": f1_score(yi, oof.argmax(1), average="macro")}; pred = te.argmax(1)
    else:
        def stack(key): a = np.zeros(len(y)); [a.__setitem__(r["va_idx"], r[key]) for r in R]; return a
        va_a, va_p, va_r = stack("va_a"), stack("va_p"), stack("va_r")
        te_a = np.mean([r["te_a"] for r in R], 0); te_p = np.mean([r["te_p"] for r in R], 0); te_r = np.mean([r["te_r"] for r in R], 0)
        res["NMAE_zero"] = nmae(y, np.zeros_like(y)); res["NMAE_mae"] = nmae(y, np.clip(va_a, 0, None))
        ths = np.arange(0.1, 0.95, 0.025)
        sc = [nmae(y, np.where(va_p >= t, np.clip(va_r, 0, None), 0)) for t in ths]; bt = float(ths[int(np.argmax(sc))])
        res["two_stage_thr"] = bt; res["NMAE_two_stage"] = max(sc)
        if res["NMAE_two_stage"] > res["NMAE_mae"]: pred = np.where(te_p >= bt, np.clip(te_r, 0, None), 0); res["use"] = "two_stage"
        else: pred = np.clip(te_a, 0, None); res["use"] = "mae"
    res = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}
    print(task, json.dumps(res, ensure_ascii=False))
    os.makedirs(os.path.join(HERE, "preds"), exist_ok=True)
    pd.DataFrame({"pred": pred}).to_csv(os.path.join(HERE, "preds", f"{task}_hgb.csv"), index=False)   # ลำดับแถว = test.csv
    json.dump(res, open(os.path.join(HERE, "preds", f"{task}_hgb_cv.json"), "w"), ensure_ascii=False)


if __name__ == "__main__":
    task, rest = sys.argv[1], sys.argv[2:]
    if rest == ["report"]: report(task)
    else:
        for k in (map(int, rest) if rest else range(NF)): fit_fold(task, k)
