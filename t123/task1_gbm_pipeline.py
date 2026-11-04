"""
Task 1 (Anti-Cheat, F2 macro) — OOP pipeline: preprocessing + feature engineering + XGBoost / LightGBM / CatBoost
+ Optuna tuning (up to 200 trials per model, time-capped) + 5-fold OOF + optimised weight blending + threshold tuning.

Run on Kaggle: GPU T4 (XGB / CatBoost use it), Internet not required, competition data attached.
NO SUBMIT — writes /kaggle/working/t1_pipeline/ only.

Design notes (from EDA):
  * 16 of 31 features carry no signal (AUC 0.500) -> dropped by default (DROP_NOISE).
  * Class-exclusive ranges (reports==4 cheater only, >=20 legit only, crosshair <0.82 / >0.94, level <28 / >96,
    age <15 or >1000 legit only, friends <10 or >269 legit only) -> explicit zone flags.
  * Hidden identity: level_progression_speed = level * r / account_age_days with r ~ U(0.15, 1.08)
    -> bounds/imputation for missing age or level.
  * Account age forms 5 segments with different cheater rates -> categorical segment.
  * Optuna objective = mean OOF AUC on 3-fold (threshold-free, smooth); F2 macro threshold is tuned at the end.
"""
import os, time, json, glob, warnings
from dataclasses import dataclass, field
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import optuna, xgboost as xgb, lightgbm as lgb, catboost as cb
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, fbeta_score
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================== config ==============================
@dataclass
class Config:
    data_root: str = os.environ.get("DATA_ROOT", "/kaggle/input")
    out_dir: str = os.environ.get("OUT_DIR", "/kaggle/working/t1_pipeline")
    target: str = "is_cheater"
    seed: int = 42
    n_folds: int = 5                    # final OOF
    tune_folds: int = 3                 # folds used inside each Optuna trial (speed)
    n_trials: int = int(os.environ.get("N_TRIALS", 200))
    tune_timeout_s: int = int(os.environ.get("TUNE_TIMEOUT", 2 * 3600 + 1200))   # per model; Kaggle session cap ~12h
    tune_lr: float = 0.05               # fixed LR while tuning (fast); final LR below
    final_lr: float = 0.02
    max_rounds: int = 20000
    early_stop: int = 200
    models: tuple = ("xgb", "lgb", "cat")
    drop_noise: bool = True
    gpu: bool = field(default_factory=lambda: os.system("nvidia-smi > /dev/null 2>&1") == 0)

    SIGNAL = ["kill_death_ratio", "headshot_percentage", "win_rate", "accuracy_score", "kill_consistency",
              "reaction_time_ms", "account_age_days", "level", "level_progression_speed", "friend_network_size",
              "reports_received", "aiming_smoothness", "spray_control_score", "game_sense_score", "crosshair_placement"]


# ============================== data ==============================
class DataLoader:
    def __init__(self, cfg: Config): self.cfg = cfg
    def _find(self, name, must=None):
        for _ in range(30):
            for d, _, fs in os.walk(self.cfg.data_root, followlinks=True):
                if name in fs and (must is None or must in d): return os.path.join(d, name)
            time.sleep(10)
        return None
    def load(self):
        tr, te = self._find("train.csv", "01_binary"), self._find("test.csv", "01_binary")
        assert tr and te, "task 1 data not found"
        sample = self._find("sample_submission.csv")
        train, test = pd.read_csv(tr), pd.read_csv(te)
        unlabeled = train[self.cfg.target].isna().sum()
        train = train[train[self.cfg.target].notna()].reset_index(drop=True)   # 12.7% unlabeled -> dropped (MCAR)
        ids = pd.read_csv(sample)["id"].values if sample else np.arange(len(test))
        print(f"train {train.shape} (dropped {unlabeled} unlabeled) | test {test.shape} | cheater rate {train[self.cfg.target].mean():.3f}")
        return train, test, ids


# ============================== preprocessing + feature engineering ==============================
class FeatureEngineer:
    """fit() learns statistics on TRAIN only (no test leakage); transform() applies to any frame."""
    CAT_COLS = ["reports_cat", "age_segment", "friends_bin", "level_bin", "crosshair_bin"]

    def __init__(self, cfg: Config): self.cfg = cfg; self.stats = {}

    def fit(self, df: pd.DataFrame):
        base = self._base(df)
        self.stats["skill_cols"] = ["kill_death_ratio", "headshot_percentage", "accuracy_score", "win_rate",
                                    "kill_consistency", "aiming_smoothness", "spray_control_score", "game_sense_score",
                                    "crosshair_placement"]
        self.stats["mu"] = base[self.stats["skill_cols"]].median()
        self.stats["sd"] = base[self.stats["skill_cols"]].std() + 1e-9
        for c, nb in [("friend_network_size", 24), ("level", 20), ("crosshair_placement", 20)]:
            self.stats[f"edges_{c}"] = np.unique(np.nanquantile(df[c], np.linspace(0, 1, nb + 1)))
        self.stats["freq_reports"] = df["reports_received"].value_counts(normalize=True).to_dict()
        self.stats["freq_friends"] = df["friend_network_size"].value_counts(normalize=True).to_dict()
        return self

    def _base(self, df):
        cols = self.cfg.SIGNAL if self.cfg.drop_noise else [c for c in df.columns if c not in ("player_id", self.cfg.target)]
        return df[cols].copy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._base(df)
        lv, ag, lp = df["level"], df["account_age_days"], df["level_progression_speed"]
        # --- missing values: indicators + count (trees route NaN natively; flags let them use "missingness")
        for c in ["reports_received", "friend_network_size", "account_age_days", "crosshair_placement", "level"]:
            X[f"na_{c}"] = df[c].isna().astype(np.int8)
        X["n_missing"] = df[self.cfg.SIGNAL].isna().sum(axis=1)
        # --- hidden identity lps = level * r / age, r in [0.15, 1.08] -> bounds & imputation
        X["age_lo"] = lv * 0.15 / lp; X["age_hi"] = lv * 1.08 / lp
        X["age_filled"] = ag.fillna(np.sqrt(X["age_lo"] * X["age_hi"]))
        X["lv_lo"] = lp * ag / 1.08; X["lv_hi"] = lp * ag / 0.15
        X["lvl_filled"] = lv.fillna(np.sqrt(X["lv_lo"] * X["lv_hi"]))
        X["r_hat"] = lp * ag / lv                                  # should lie in [0.15, 1.08]
        # --- class-exclusive zones (generator artefacts)
        rep, ch, age, fr = df["reports_received"], df["crosshair_placement"], X["age_filled"], df["friend_network_size"]
        X["zone_cheat"] = ((rep == 4) | (ch < 0.82) | (X["lvl_filled"] < 28)).astype(np.int8)
        X["zone_legit"] = ((rep >= 20) | (ch > 0.94) | (X["lvl_filled"] > 96) | (age > 1000) | (age < 15) | (fr < 10) | (fr > 269)).astype(np.int8)
        # --- transforms (help linear-ish splits / CatBoost borders; harmless for trees)
        for c in ["account_age_days", "friend_network_size", "level_progression_speed", "reports_received"]:
            X[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
        # --- interactions / ratios
        X["kd_x_hs"] = df["kill_death_ratio"] * df["headshot_percentage"]
        X["acc_x_cross"] = df["accuracy_score"] * df["crosshair_placement"]
        X["reports_per_100d"] = 100 * rep / (age + 1)
        X["friends_per_100d"] = 100 * fr / (age + 1)
        X["level_per_100d"] = 100 * X["lvl_filled"] / (age + 1)
        Z = (df[self.stats["skill_cols"]] - self.stats["mu"]) / self.stats["sd"]
        X["skill_z_sum"] = Z.sum(axis=1, min_count=3); X["skill_z_std"] = Z.std(axis=1)
        # --- frequency encoding (fit on train)
        X["freq_reports"] = rep.map(self.stats["freq_reports"]).astype(float)
        X["freq_friends"] = fr.map(self.stats["freq_friends"]).astype(float)
        # --- categorical features (numeric-as-categorical + segments)
        X["reports_cat"] = rep.fillna(-1).astype(int).astype(str)
        X["age_segment"] = pd.cut(age, [-np.inf, 15, 100, 200, 1000, np.inf], labels=["a<15", "a15-100", "a100-200", "a200-1000", "a>1000"]).astype(str)
        for c, nm in [("friend_network_size", "friends_bin"), ("level", "level_bin"), ("crosshair_placement", "crosshair_bin")]:
            e = self.stats[f"edges_{c}"]
            X[nm] = pd.cut(df[c], e, include_lowest=True, labels=False).fillna(-1).astype(int).astype(str)
        X = X.replace([np.inf, -np.inf], np.nan)
        for c in self.CAT_COLS: X[c] = X[c].astype("category")
        return X

    def align_categories(self, A, B):
        for c in self.CAT_COLS:
            sa, sb = A[c].astype(object).map(str), B[c].astype(object).map(str)
            cats = sorted(set(sa) | set(sb))
            A[c] = pd.Categorical(sa, categories=cats); B[c] = pd.Categorical(sb, categories=cats)
        return A, B


# ============================== models ==============================
class BaseModel:
    name = "base"
    def __init__(self, cfg: Config, params: dict, lr: float): self.cfg, self.params, self.lr = cfg, params, lr
    def fit(self, Xa, ya, Xv, yv): raise NotImplementedError
    def predict(self, X): raise NotImplementedError
    @staticmethod
    def suggest(trial): raise NotImplementedError

class XGBModel(BaseModel):
    name = "xgb"
    @staticmethod
    def suggest(t):
        return dict(max_depth=t.suggest_int("max_depth", 3, 10), min_child_weight=t.suggest_float("min_child_weight", 1, 100, log=True),
                    subsample=t.suggest_float("subsample", 0.5, 1.0), colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 50, log=True), reg_alpha=t.suggest_float("reg_alpha", 1e-3, 20, log=True),
                    gamma=t.suggest_float("gamma", 1e-4, 5, log=True), max_bin=t.suggest_categorical("max_bin", [128, 256, 512]),
                    grow_policy=t.suggest_categorical("grow_policy", ["depthwise", "lossguide"]))
    def fit(self, Xa, ya, Xv, yv):
        self.m = xgb.XGBClassifier(n_estimators=self.cfg.max_rounds, learning_rate=self.lr, early_stopping_rounds=self.cfg.early_stop,
                                   eval_metric="auc", tree_method="hist", device="cuda" if self.cfg.gpu else "cpu",
                                   enable_categorical=True, max_cat_to_onehot=8, random_state=self.cfg.seed, **self.params)
        self.m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False); self.best_iter = self.m.best_iteration; return self
    def predict(self, X): return self.m.predict_proba(X)[:, 1]

class LGBModel(BaseModel):
    name = "lgb"
    @staticmethod
    def suggest(t):
        return dict(num_leaves=t.suggest_int("num_leaves", 7, 255, log=True), min_child_samples=t.suggest_int("min_child_samples", 5, 300, log=True),
                    subsample=t.suggest_float("subsample", 0.5, 1.0), subsample_freq=1, colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 50, log=True), reg_alpha=t.suggest_float("reg_alpha", 1e-3, 20, log=True),
                    min_split_gain=t.suggest_float("min_split_gain", 1e-4, 1, log=True), max_bin=t.suggest_categorical("max_bin", [63, 127, 255, 511]),
                    cat_smooth=t.suggest_float("cat_smooth", 1, 100, log=True), cat_l2=t.suggest_float("cat_l2", 1, 50, log=True))
    def fit(self, Xa, ya, Xv, yv):
        self.m = lgb.LGBMClassifier(n_estimators=self.cfg.max_rounds, learning_rate=self.lr, random_state=self.cfg.seed, verbose=-1, n_jobs=-1, **self.params)
        self.m.fit(Xa, ya, eval_set=[(Xv, yv)], eval_metric="auc", callbacks=[lgb.early_stopping(self.cfg.early_stop, verbose=False)])
        self.best_iter = self.m.best_iteration_; return self
    def predict(self, X): return self.m.predict_proba(X)[:, 1]

class CatModel(BaseModel):
    name = "cat"
    @staticmethod
    def suggest(t):
        p = dict(depth=t.suggest_int("depth", 4, 10), l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.5, 50, log=True),
                 random_strength=t.suggest_float("random_strength", 1e-3, 10, log=True), border_count=t.suggest_categorical("border_count", [128, 254]),
                 bootstrap_type=t.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]))
        if p["bootstrap_type"] == "Bayesian": p["bagging_temperature"] = t.suggest_float("bagging_temperature", 0, 5)
        else: p["subsample"] = t.suggest_float("subsample", 0.5, 1.0)
        return p
    def fit(self, Xa, ya, Xv, yv):
        cats = [c for c in Xa.columns if str(Xa[c].dtype) == "category"]
        conv = lambda D: D.assign(**{c: D[c].astype(object).map(str) for c in cats})
        self.cats, self.conv = cats, conv
        self.m = cb.CatBoostClassifier(iterations=self.cfg.max_rounds, learning_rate=self.lr, eval_metric="AUC", od_type="Iter", od_wait=self.cfg.early_stop,
                                       task_type="GPU" if self.cfg.gpu else "CPU", random_seed=self.cfg.seed, verbose=False, **self.params)
        self.m.fit(conv(Xa), ya, cat_features=cats, eval_set=(conv(Xv), yv), use_best_model=True); self.best_iter = self.m.get_best_iteration(); return self
    def predict(self, X): return self.m.predict_proba(self.conv(X))[:, 1]

MODELS = {"xgb": XGBModel, "lgb": LGBModel, "cat": CatModel}


# ============================== CV + tuning ==============================
class CVRunner:
    """K-fold; each fold carves an inner 10% holdout for early stopping so OOF stays clean."""
    def __init__(self, cfg, X, y, Xt=None): self.cfg, self.X, self.y, self.Xt = cfg, X, y, Xt
    def run(self, cls, params, lr, n_folds, seed_offset=0, verbose=False):
        folds = StratifiedKFold(n_folds, shuffle=True, random_state=self.cfg.seed).split(self.X, self.y)
        oof = np.full(len(self.y), np.nan); tp = np.zeros(len(self.Xt)) if self.Xt is not None else None; iters = []
        for k, (a, b) in enumerate(folds):
            rng = np.random.default_rng(self.cfg.seed + k + seed_offset); a = rng.permutation(a); cut = int(0.9 * len(a))
            m = cls(self.cfg, params, lr).fit(self.X.iloc[a[:cut]], self.y[a[:cut]], self.X.iloc[a[cut:]], self.y[a[cut:]])
            oof[b] = m.predict(self.X.iloc[b]); iters.append(m.best_iter)
            if tp is not None: tp += m.predict(self.Xt) / n_folds
            if verbose: print(f"   fold {k}: AUC {roc_auc_score(self.y[b], oof[b]):.4f} | best_iter {m.best_iter}", flush=True)
        return oof, tp, iters

class OptunaTuner:
    def __init__(self, cfg, runner): self.cfg, self.runner = cfg, runner
    def tune(self, name):
        cls = MODELS[name]
        def objective(trial):
            oof, _, _ = self.runner.run(cls, cls.suggest(trial), self.cfg.tune_lr, self.cfg.tune_folds)
            return roc_auc_score(self.runner.y, oof)
        storage = f"sqlite:///{self.cfg.out_dir}/optuna_{name}.db"              # resumable; partial results survive
        study = optuna.create_study(direction="maximize", study_name=f"t1_{name}", storage=storage, load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=self.cfg.seed, multivariate=True, n_startup_trials=25))
        t0 = time.time(); done0 = len(study.trials)
        def cb_(st, tr):
            n = len(st.trials) - done0
            if n % 25 == 0: print(f"   [{name}] trial {len(st.trials)} | best AUC {st.best_value:.5f} | {(time.time()-t0)/60:.0f} min", flush=True)
        study.optimize(objective, n_trials=max(0, self.cfg.n_trials - done0), timeout=self.cfg.tune_timeout_s, callbacks=[cb_], gc_after_trial=True)
        print(f"[{name}] tuning done: {len(study.trials)} trials | best 3-fold AUC {study.best_value:.5f} | {(time.time()-t0)/60:.0f} min")
        print(f"[{name}] best params: {study.best_params}", flush=True)
        params = dict(study.best_params)
        if name == "lgb": params["subsample_freq"] = 1
        return params, study.best_value, len(study.trials)


# ============================== blending + threshold ==============================
class Blender:
    """Weights on the simplex (softmax-parametrised) maximising OOF AUC on rank-normalised predictions,
    evaluated with an outer CV so the reported blend score is not optimistic."""
    def __init__(self, y, seed=42): self.y, self.seed = y, seed
    @staticmethod
    def _rank(P): return np.column_stack([rankdata(p) / len(p) for p in P.T])
    def _fit(self, R, y):
        k = R.shape[1]; f = lambda z: -roc_auc_score(y, R @ (np.exp(z) / np.exp(z).sum()))
        best = min((minimize(f, z0, method="Nelder-Mead", options=dict(maxiter=400, xatol=1e-4, fatol=1e-6))
                    for z0 in [np.zeros(k)] + [np.random.default_rng(i).normal(0, 1, k) for i in range(4)]), key=lambda r: r.fun)
        w = np.exp(best.x); return w / w.sum()
    def fit(self, P_oof):
        R = self._rank(P_oof); self.w = self._fit(R, self.y)
        cv = np.zeros(len(self.y))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=self.seed + 1).split(R, self.y):
            cv[b] = R[b] @ self._fit(R[a], self.y[a])
        self.cv_blend = cv; return self
    def blend(self, P): return self._rank(P) @ self.w

class ThresholdOptimizer:
    @staticmethod
    def f2m(y, p, t): return fbeta_score(y, (p >= t).astype(int), beta=2, average="macro")
    def fit(self, y, p):
        grid = np.quantile(p, np.linspace(0.30, 0.80, 251))          # rank-scale blends -> search on quantiles
        sc = [self.f2m(y, p, t) for t in grid]; i = int(np.argmax(sc))
        self.t, self.score, self.pos_rate = grid[i], sc[i], float((p >= grid[i]).mean()); return self


# ============================== orchestration ==============================
class Pipeline:
    def __init__(self, cfg: Config): self.cfg = cfg; os.makedirs(cfg.out_dir, exist_ok=True)
    def run(self):
        T0 = time.time(); cfg = self.cfg
        print(f"GPU: {cfg.gpu} | models {cfg.models} | trials {cfg.n_trials} | timeout/model {cfg.tune_timeout_s/3600:.1f} h", flush=True)
        train, test, ids = DataLoader(cfg).load(); y = train[cfg.target].astype(int).values
        fe = FeatureEngineer(cfg).fit(train); X, Xt = fe.transform(train), fe.transform(test); X, Xt = fe.align_categories(X, Xt)
        print(f"features: {X.shape[1]} ({len(fe.CAT_COLS)} categorical)", flush=True)
        runner = CVRunner(cfg, X, y, Xt); tuner = OptunaTuner(cfg, runner)
        res, P_oof, P_te = {}, [], []
        for name in cfg.models:
            print(f"\n===== {name}: Optuna =====", flush=True)
            params, tune_auc, ntr = tuner.tune(name)
            print(f"===== {name}: final {cfg.n_folds}-fold @ lr {cfg.final_lr} =====", flush=True)
            oof, tp, iters = runner.run(MODELS[name], params, cfg.final_lr, cfg.n_folds, verbose=True)
            th = ThresholdOptimizer().fit(y, oof)
            res[name] = dict(trials=ntr, tune_auc_3fold=round(tune_auc, 5), oof_auc=round(roc_auc_score(y, oof), 5),
                             oof_f2m=round(th.score, 5), thr=round(float(th.t), 4), best_iters=iters, params=params)
            print(f"[{name}] OOF AUC {res[name]['oof_auc']} | F2 macro {res[name]['oof_f2m']} @ thr {th.t:.3f}", flush=True)
            P_oof.append(oof); P_te.append(tp)
            np.save(f"{cfg.out_dir}/oof_{name}.npy", oof); np.save(f"{cfg.out_dir}/test_{name}.npy", tp)
            json.dump(res, open(f"{cfg.out_dir}/results.json", "w"), indent=1, default=str)
        P_oof, P_te = np.column_stack(P_oof), np.column_stack(P_te)
        bl = Blender(y, cfg.seed).fit(P_oof); oof_b, te_b = bl.blend(P_oof), bl.blend(P_te)
        th = ThresholdOptimizer().fit(y, oof_b); th_cv = ThresholdOptimizer().fit(y, bl.cv_blend)
        res["blend"] = dict(weights=dict(zip(cfg.models, np.round(bl.w, 4).tolist())), oof_auc=round(roc_auc_score(y, oof_b), 5),
                            oof_f2m=round(th.score, 5), cv_blend_auc=round(roc_auc_score(y, bl.cv_blend), 5), cv_blend_f2m=round(th_cv.score, 5),
                            thr_rank=round(float(th.t), 4), pred_pos_rate=round(th.pos_rate, 4))
        print("\n================ SUMMARY ================")
        print("reference (earlier): HGB 0.7782 | HGB+age bounds 0.7799 | TabPFN+HGB 0.7825")
        for k, v in res.items(): print(k, {kk: vv for kk, vv in v.items() if kk not in ("params", "best_iters")})
        # test predictions: same rank scale as OOF -> use the predicted-positive rate learned on OOF
        cut = np.quantile(te_b, 1 - th.pos_rate)
        pred = (te_b >= cut).astype(int)
        pd.DataFrame({"id": ids, "pred": pred, "score": te_b}).to_csv(f"{cfg.out_dir}/t1_blend.csv", index=False)
        np.save(f"{cfg.out_dir}/oof_blend.npy", oof_b)
        json.dump(res, open(f"{cfg.out_dir}/results.json", "w"), indent=1, default=str)
        print(f"test predicted cheater rate {pred.mean():.3f} (OOF {th.pos_rate:.3f}) | total {(time.time()-T0)/3600:.2f} h")
        print("wrote", cfg.out_dir, "- nothing submitted")


if __name__ == "__main__":
    cfg = Config()
    if os.environ.get("SMOKE") == "1":
        cfg.n_trials, cfg.tune_timeout_s, cfg.max_rounds, cfg.tune_lr, cfg.final_lr, cfg.early_stop = 3, 60, 200, 0.2, 0.2, 20
    Pipeline(cfg).run()
