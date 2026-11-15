"""
Task 5 — Temporal Masked MLP (TMLP): เวอร์ชัน scikit-learn ของ TMAE v2 (ไม่ต้องใช้ PyTorch)

หลักการเดียวกับ TMAE v2:
  - "p4"  : ทายช่วง 4 จากช่วง 1–3 (+static)   → จับขโมยบัญชี / การกระโดดช่วงสุดท้าย
  - "p34" : ทายช่วง 3+4 จากช่วง 1–2 (+static) → จับ VPN ที่กระโดดช่วง 3 แล้วอยู่ต่อ
  ช่องที่ว่างส่งเป็น mask (0/1) เข้าโมเดล, iterative trimming ตัดแถว error สูงสุด 5% แล้ว fit ใหม่
  คะแนน = top-k robust z ของ error (ไม่นับช่องที่ไม่มีค่า) → rank → max ของ 2 ชุด → ensemble กับ IF-jump

รันทีละขั้น (แต่ละขั้นจบใน ~1–3 นาที):
  python task5_tmlp.py prep
  python task5_tmlp.py fit p4 0      # round 0 (ใช้ทุกแถว)
  python task5_tmlp.py fit p4 1      # round 1 (ตัด 5% ที่ error สูงสุดจาก round 0)
  python task5_tmlp.py fit p34 0
  python task5_tmlp.py fit p34 1
  python task5_tmlp.py score
ไม่ submit อะไร — เขียนไฟล์ลง preds/ และ probes/ เท่านั้น
"""
import os, sys, glob, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "tmlp_work"); os.makedirs(WORK, exist_ok=True)
SEED, CLIP_Z, TOPK, MAD_FLOOR, TRIM_FRAC = 42, 20.0, 3, 0.05, 0.05
HIDDEN = (128, 64)
FLAG_RATES = [0.035, 0.05, 0.08]
SETS = {"p4": ([1, 2, 3], [4]), "p34": ([1, 2], [3, 4])}   # (input periods, target periods)


def find(name, must=None):
    hits = glob.glob(os.path.join(HERE, "..", "**", name), recursive=True)
    if must: hits = [h for h in hits if any(m in h for m in must)]
    return sorted(hits)[0]


def prep():
    test = find("test.csv", ["05_anomaly"]); sample = find("sample_submission.csv")
    raw = pd.read_csv(test); ids = pd.read_csv(sample)["id"]; assert len(raw) == len(ids)
    X = raw.drop(columns=["player_id"]); P = [1, 2, 3, 4]
    bases = sorted({c.rsplit("_", 1)[0] for c in X.columns if c[-2:] in ("_1", "_2", "_3", "_4")})
    STATIC = [c for c in X.columns if c[-2:] not in ("_1", "_2", "_3", "_4")]
    NONGEO = [b for b in bases if b not in ("login_lat", "login_lon")]
    w = lambda x: (x + 180.0) % 360.0 - 180.0
    lat = np.stack([X[f"login_lat_{p}"].values for p in P], 1); lon = np.stack([X[f"login_lon_{p}"].values for p in P], 1)
    disp = lambda a1, o1, a2, o2: np.hypot(a2 - a1, w(o2 - o1))
    step = np.full_like(lat, np.nan); home = np.full_like(lat, np.nan)
    for i in range(1, 4):
        step[:, i] = disp(lat[:, i-1], lon[:, i-1], lat[:, i], lon[:, i])
        h_lat = np.nanmean(lat[:, :i], 1)
        h_lon = np.degrees(np.arctan2(np.nanmean(np.sin(np.radians(lon[:, :i])), 1), np.nanmean(np.cos(np.radians(lon[:, :i])), 1)))
        home[:, i] = disp(h_lat, h_lon, lat[:, i], lon[:, i])
    inv = np.where(np.isnan(lat), np.nan, (np.abs(lat) > 90).astype(float))
    ch = {b: np.stack([X[f"{b}_{p}"].values for p in P], 1) for b in NONGEO}
    ch.update(lat=lat, lon_sin=np.sin(np.radians(lon)), lon_cos=np.cos(np.radians(lon)),
              log_step=np.log1p(step), log_disp_home=np.log1p(home), invalid_lat=inv)
    CH = list(ch); XP = np.stack([ch[c] for c in CH], -1).astype(np.float32); XS = X[STATIC].values.astype(np.float32)

    def rfit(a):
        a = a[~np.isnan(a)]; med = np.median(a); q1, q3 = np.percentile(a, [25, 75]); iqr = q3 - q1
        return med, (iqr if iqr > 1e-8 else (a.std() if a.std() > 1e-8 else 1.0))
    MP = (~np.isnan(XP)).astype(np.float32); MS = (~np.isnan(XS)).astype(np.float32)
    for c in range(XP.shape[2]):
        m, s = rfit(XP[:, :, c].ravel()); XP[:, :, c] = np.clip((XP[:, :, c] - m) / s, -CLIP_Z, CLIP_Z)
    for c in range(XS.shape[1]):
        m, s = rfit(XS[:, c]); XS[:, c] = np.clip((XS[:, c] - m) / s, -CLIP_Z, CLIP_Z)

    # IF-jump features (สำหรับ ensemble) + proxy (สำหรับ sanity check)
    Fj = {}
    for b in NONGEO:
        v = np.stack([X[f"{b}_{p}"].values for p in P], 1)
        for k in (3, 4):
            j = v[:, k-1] - np.nanmean(v[:, :k-1], 1); m = np.nanmedian(j); s = 1.4826 * np.nanmedian(np.abs(j - m)) + 1e-9
            Fj[f"{b}_jz{k}"] = np.abs((j - m) / s)
    for k in (2, 3, 4): Fj[f"geo_disp{k}"] = np.log1p(home[:, k-1])
    Fj["invalid_lat"] = np.nanmax(np.nan_to_num(inv), 1)
    for c in STATIC: Fj[c] = X[c].values
    Zj = pd.DataFrame(Fj).fillna(0).clip(upper=50)
    shock = np.stack([Fj[f"{b}_jz4"] > 6 for b in NONGEO], 1).sum(1)
    g3, g4 = np.nan_to_num(home[:, 2]) > 5, np.nan_to_num(home[:, 3]) > 5
    np.savez_compressed(os.path.join(WORK, "data.npz"), XP=np.nan_to_num(XP), MP=MP, XS=np.nan_to_num(XS), MS=MS,
                        Zj=Zj.values.astype(np.float32), shock=shock, g3=g3, g4=g4, inv=Fj["invalid_lat"],
                        pw=X["password_changed"].values, ids=np.array(ids.tolist(), dtype="U16"), CH=np.array(CH, dtype="U40"))
    print(f"prep done: XP {XP.shape}, static {XS.shape}, IF feats {Zj.shape}")


def load():
    return np.load(os.path.join(WORK, "data.npz"), allow_pickle=False)


def design(d, name):
    pin, pout = SETS[name]; ii = [p - 1 for p in pin]; oo = [p - 1 for p in pout]
    XP, MP = d["XP"], d["MP"]; N = len(XP)
    Xin = np.concatenate([XP[:, ii].reshape(N, -1), MP[:, ii].reshape(N, -1), d["XS"], d["MS"]], 1)
    # target ที่ว่าง → เติมด้วยค่าเฉลี่ยของช่วง input ที่มีค่า (กันโมเดลเรียนทาย 0 มั่ว ๆ); ตอนให้คะแนนไม่นับช่องนี้
    seen = (XP[:, ii] * MP[:, ii]).sum(1) / np.maximum(MP[:, ii].sum(1), 1)
    Y = XP[:, oo].copy(); M = MP[:, oo]
    Y = np.where(M > 0, Y, seen[:, None, :]).reshape(N, -1); M = M.reshape(N, -1)
    return Xin.astype(np.float32), Y.astype(np.float32), M


def fit(name, rnd):
    from sklearn.neural_network import MLPRegressor
    import joblib
    d = load(); Xin, Y, M = design(d, name); N = len(Xin)
    keep = np.ones(N, bool)
    if rnd > 0:
        prev = np.load(os.path.join(WORK, f"err_{name}_r{rnd-1}.npy"))
        keep = prev <= np.quantile(prev, 1 - TRIM_FRAC)
    t0 = time.time()
    mdl = MLPRegressor(hidden_layer_sizes=HIDDEN, activation="relu", alpha=1e-4, batch_size=256,
                       learning_rate_init=1e-3, max_iter=60, early_stopping=True, n_iter_no_change=6,
                       random_state=SEED + rnd)
    mdl.fit(Xin[keep], Y[keep])
    err = np.abs(mdl.predict(Xin) - Y)
    per_row = (err * M).sum(1) / np.maximum(M.sum(1), 1)      # ใช้คัดแถวสำหรับ round ถัดไป
    np.save(os.path.join(WORK, f"err_{name}_r{rnd}.npy"), per_row)
    np.save(os.path.join(WORK, f"abserr_{name}_r{rnd}.npy"), np.where(M > 0, err, np.nan).astype(np.float32))
    joblib.dump(mdl, os.path.join(WORK, f"mlp_{name}_r{rnd}.joblib"))
    print(f"fit {name} round {rnd}: trained on {keep.sum()} rows, {mdl.n_iter_} iters, "
          f"best val R2 {max(mdl.validation_scores_):.3f}, {time.time()-t0:.0f}s")


def topk_score(err, k=TOPK):
    med = np.nanmedian(err, 0); mad = np.maximum(np.nanmedian(np.abs(err - med), 0) * 1.4826, MAD_FLOOR)
    z = np.clip((err - med) / mad, 0, None); z = np.where(np.isnan(z), -np.inf, z)
    top = -np.sort(-z, 1)[:, :k]; return np.where(np.isinf(top), 0, top).mean(1), z


def score():
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score
    from scipy.stats import rankdata
    d = load(); ids = d["ids"]; CH = list(d["CH"]); N = len(ids)
    rank = lambda s: rankdata(s) / len(s)
    last = {n: max(int(f.split("_r")[-1][0]) for f in os.listdir(WORK) if f.startswith(f"abserr_{n}_r")) for n in SETS}
    S, Z = {}, {}
    for n in SETS:
        S[n], Z[n] = topk_score(np.load(os.path.join(WORK, f"abserr_{n}_r{last[n]}.npy")))
    tmlp = np.max(np.stack([rank(S[n]) for n in SETS], 1), 1)
    iff = -IsolationForest(n_estimators=500, max_samples=1024, random_state=SEED, n_jobs=-1).fit(d["Zj"]).score_samples(d["Zj"])
    final = (rank(tmlp) + rank(iff)) / 2

    g3, g4, inv, shock = d["g3"], d["g4"], d["inv"], d["shock"]
    proxy = g3 | g4 | (inv > 0) | (shock >= 2)
    types = {"VPN geo (p3)": g3, "theft geo (p4 only)": g4 & ~g3, "invalid lat": inv > 0, "shock>=2 no geo": (shock >= 2) & ~g3 & ~g4}
    print(f"rounds used: {last} | proxy-obvious rows: {proxy.sum()} (sanity only, not ground truth)")
    for name, s in [("TMLP", tmlp), ("IF-jump", iff), ("Ensemble", final)]:
        rec = {f"{r:.1%}": round(((s >= np.quantile(s, 1 - r)) & proxy).sum() / proxy.sum(), 3) for r in FLAG_RATES}
        top = s >= np.quantile(s, 0.95)
        print(f"  {name:9s} recall {rec} | @5% by type", {k: f"{(top & m).sum()}/{m.sum()}" for k, m in types.items()})
    pw = d["pw"]; ok = ~np.isnan(pw)
    print("  AUC(TMLP vs password_changed) =", round(roc_auc_score(pw[ok], tmlp[ok]), 3), "(ควรห่างจาก 1.0)")
    print("  Spearman(TMLP, IF) =", round(pd.Series(tmlp).corr(pd.Series(iff), method="spearman"), 3))

    # อธิบาย top 10
    def explain(i):
        items = []
        for n, (_, pout) in SETS.items():
            for j, z in enumerate(Z[n][i]):
                if np.isfinite(z): items.append((z, f"[{n}] p{pout[j // len(CH)]}:{CH[j % len(CH)]}"))
        return ", ".join(f"{t}({z:.0f})" for z, t in sorted(items, reverse=True)[:3])
    print("\nTop 10 (ensemble):")
    for i in np.argsort(-final)[:10]:
        print(f"  {ids[i]}  {final[i]:.3f}  {explain(i)}")

    out = os.path.join(HERE, "preds"); os.makedirs(out, exist_ok=True); os.makedirs(os.path.join(HERE, "probes"), exist_ok=True)
    pd.DataFrame({"id": ids, "tmlp": tmlp, "tmlp_p4": S["p4"], "tmlp_p34": S["p34"], "iforest": iff, "final": final}) \
        .to_csv(os.path.join(out, "task5_tmlp_scores.csv"), index=False)
    for r in FLAG_RATES:
        pred = (final >= np.quantile(final, 1 - r)).astype(int); tag = f"tmlp_ens_c{r*1000:03.0f}"
        pd.DataFrame({"id": ids, "pred": pred}).to_csv(os.path.join(out, f"task5_{tag}.csv"), index=False)
        pd.DataFrame({"id": ids, "task1": 2, "task2": 9, "task3": 0, "task4": 9, "task5": pred}) \
            .to_csv(os.path.join(HERE, "probes", f"P5_task5_only_mask_{tag}.csv"), index=False)
    print("\nเขียน preds/ และ probes/ แล้ว — ไม่ได้ submit อะไร")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep": prep()
    elif cmd == "fit": fit(sys.argv[2], int(sys.argv[3]))
    elif cmd == "score": score()
