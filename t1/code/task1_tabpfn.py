# Task 1 — TabPFN v3.5 via tabpfn-client (Prior Labs cloud API) vs HGB baseline, F2 macro
# Needs: Internet ON + Kaggle secret TABPFN_TOKEN (Add-ons -> Secrets). Token is never written in code.
# NOTE: tabpfn-client sends the training/test feature rows to Prior Labs servers for inference.
# Same 5 folds as GBM (StratifiedKFold(5, shuffle, 42) on labeled rows). NO SUBMIT - writes /kaggle/working/preds only
import os, sys, time, json, subprocess, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "tabpfn-client"], check=False)
import tabpfn_client
from tabpfn_client import TabPFNClassifier
from kaggle_secrets import UserSecretsClient
tabpfn_client.set_access_token(UserSecretsClient().get_secret("TABPFN_TOKEN"))
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

SEED, NF = 42, 5
MODEL = os.environ.get("MODEL", "v3.5_default")
N_EST = int(os.environ.get("N_EST", 8))
CTX = int(os.environ.get("CTX", 20000))       # context rows per fit (lower this if the API rejects the request size)
BATCH = int(os.environ.get("BATCH", 5000))    # rows per predict call
ROOT, OUT = "/kaggle/input", "/kaggle/working/preds"
def walk_find(name, must=None):
    for d, _, fs in os.walk(ROOT, followlinks=True):
        if name in fs and (must is None or must in d): return os.path.join(d, name)
for _ in range(30):
    TRAIN, TEST = walk_find("train.csv", "01_binary"), walk_find("test.csv", "01_binary")
    if TRAIN and TEST: break
    time.sleep(10)
SAMPLE = walk_find("sample_submission.csv")
assert TRAIN and TEST, "data not found"
tr, te = pd.read_csv(TRAIN), pd.read_csv(TEST)
L = tr[tr["is_cheater"].notna()].reset_index(drop=True); y = L["is_cheater"].astype(int).values
SIG = ["kill_death_ratio", "headshot_percentage", "win_rate", "accuracy_score", "kill_consistency", "reaction_time_ms",
       "account_age_days", "level", "level_progression_speed", "friend_network_size", "reports_received",
       "aiming_smoothness", "spray_control_score", "game_sense_score", "crosshair_placement"]
def fe(D):                                    # 15 signal features + age/level bounds (lps = level * r / age, r ~ U(0.15, 1.08))
    X = D[SIG].copy(); lv, ag, lp = D["level"], D["account_age_days"], D["level_progression_speed"]
    X["age_lo"] = lv * 0.15 / lp; X["age_hi"] = lv * 1.08 / lp
    X["age_filled"] = ag.fillna(np.sqrt(X["age_lo"] * X["age_hi"]))
    X["lv_lo"] = lp * ag / 1.08; X["lv_hi"] = lp * ag / 0.15
    return X.replace([np.inf, -np.inf], np.nan)
X, Xt = fe(L), fe(te)
ids = pd.read_csv(SAMPLE)["id"].values if SAMPLE else np.arange(len(te))
print(L.shape, te.shape, "| features", X.shape[1], "| model", MODEL, "| n_est", N_EST, "| ctx", CTX, flush=True)

def f2m_best(yt, p):
    ths = np.arange(0.1, 0.8, 0.005); sc = [fbeta_score(yt, (p >= t).astype(int), beta=2, average="macro") for t in ths]
    i = int(np.argmax(sc)); return sc[i], ths[i]
def pfn_fit(idx):
    m = TabPFNClassifier(model_path=MODEL, n_estimators=N_EST)
    ctx = idx if len(idx) <= CTX else np.sort(np.random.default_rng(SEED).choice(idx, CTX, replace=False))
    m.fit(X.iloc[ctx], y[ctx]); return m
def pfn_predict(m, Z):
    out = []
    for i in range(0, len(Z), BATCH):
        for att in range(3):
            try: out.append(m.predict_proba(Z.iloc[i:i+BATCH])[:, 1]); break
            except Exception as e:
                print("  retry", att, type(e).__name__, str(e)[:200], flush=True); time.sleep(20)
        else: raise RuntimeError("predict failed")
    return np.concatenate(out)

folds = list(StratifiedKFold(NF, shuffle=True, random_state=SEED).split(X, y))
oof_pfn, oof_hgb = np.zeros(len(y)), np.zeros(len(y))
for k, (a, b) in enumerate(folds):
    t0 = time.time()
    oof_pfn[b] = pfn_predict(pfn_fit(a), X.iloc[b])
    oof_hgb[b] = HGB(max_iter=300, learning_rate=0.03, max_leaf_nodes=31, random_state=SEED).fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
    print(f"fold {k}: {time.time()-t0:.0f}s | AUC TabPFN {roc_auc_score(y[b], oof_pfn[b]):.4f} | HGB {roc_auc_score(y[b], oof_hgb[b]):.4f}", flush=True)

R = {}
for nm, p in [("TabPFN", oof_pfn), ("HGB", oof_hgb), ("avg", (oof_pfn + oof_hgb) / 2)]:
    s, t = f2m_best(y, p); R[nm] = {"AUC": round(roc_auc_score(y, p), 4), "F2macro": round(s, 4), "thr": round(float(t), 3)}
print("reference: HGB all-31 F2m 0.7782 AUC 0.8904 | signal+age-bounds F2m 0.7807 AUC 0.8922")
for k_, v in R.items(): print(f"{k_:8s} {v}")

# test: one fit on (subsampled) all labeled rows
te_pfn = pfn_predict(pfn_fit(np.arange(len(y))), Xt)
os.makedirs(OUT, exist_ok=True)
np.save(f"{OUT}/t1_tabpfn_oof.npy", oof_pfn); np.save(f"{OUT}/t1_tabpfn_test_proba.npy", te_pfn)
pd.DataFrame({"id": ids, "pred": (te_pfn >= R["TabPFN"]["thr"]).astype(int)}).to_csv(f"{OUT}/t1_tabpfn.csv", index=False)
json.dump(R, open(f"{OUT}/t1_tabpfn_summary.json", "w"), indent=1)
print("wrote", OUT, "- nothing submitted")
