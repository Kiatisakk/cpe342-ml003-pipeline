"""
Task 5 — Temporal Masked Autoencoder (TMAE v3) for unsupervised account-anomaly detection.

Learns "normal behaviour" of an account across 4 periods (plus static profile) by masked reconstruction:
  * tokens  = 4 period tokens (29 temporal metrics + home-relative geo) + 1 static token
  * masking = random feature cells, whole periods, and whole metrics across all periods (so a metric must be
              predicted from the OTHER metrics -> stationary anomalies such as 24/7 bots become unpredictable)
  * robust training: rows with the highest reconstruction loss are down-weighted each epoch (anomalies ~5-8%)
  * scoring: deterministic masking passes (each period, each metric-group across time) -> per-cell robust z
             -> top-k mean per row
No labels used. Proxy sanity check against known rule groups (rules / bot / lowact / lowcomm / lowstab).
"""
import os, sys, time, math, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import QuantileTransformer
torch.set_num_threads(8); SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)
SRC = os.environ.get("T5_TEST", "cpe342-kaggle-main/05_anomaly_detection/test.csv")
GROUPS = os.environ.get("T5_GROUPS", "submissions/probes_t5/t5_groups.csv")
OUT = os.environ.get("OUT_DIR", "t123/t5")
EPOCHS = int(os.environ.get("EPOCHS", 60)); D = 96; TRIM = 0.08

X = pd.read_csv(SRC); P = [1, 2, 3, 4]
BASES = sorted({c.rsplit("_", 1)[0] for c in X.columns if c[-2:] in ("_1", "_2", "_3", "_4")})
MET = [b for b in BASES if b not in ("login_lat", "login_lon")]
STATIC = ["account_age_days", "behavior_stability", "password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]

# ---------- features ----------
lat = np.stack([X[f"login_lat_{p}"].values for p in P], 1); lon = np.stack([X[f"login_lon_{p}"].values for p in P], 1)
bad_lat = (np.abs(lat) > 90); lat = np.where(bad_lat, np.nan, lat)
hl = np.nanmean(lat[:, :2], 1, keepdims=True)
hn = np.degrees(np.arctan2(np.nanmean(np.sin(np.radians(lon[:, :2])), 1), np.nanmean(np.cos(np.radians(lon[:, :2])), 1)))[:, None]
dlat = lat - hl; dlon = (lon - hn + 180) % 360 - 180
geo = np.stack([dlat, dlon, np.log1p(np.hypot(dlat, dlon)), bad_lat.astype(float)], 2)          # (N,4,4)
T = np.stack([np.stack([X[f"{b}_{p}"].values for b in MET], 1) for p in P], 1).astype(float)    # (N,4,M)
N, M = len(X), len(MET)
qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=SEED)
Tq = qt.fit_transform(T.reshape(N * 4, M)).reshape(N, 4, M)                                         # unsupervised, per metric
G = np.nan_to_num((geo - np.nanmean(geo, (0, 1))) / (np.nanstd(geo, (0, 1)) + 1e-6))
S = X[STATIC].values.astype(float); S = (S - np.nanmean(S, 0)) / (np.nanstd(S, 0) + 1e-6)
seq = np.concatenate([Tq, G], 2)                                                                     # (N,4,F)
F = seq.shape[2]; obs = ~np.isnan(seq); seq = np.nan_to_num(seq); Sobs = ~np.isnan(S); S = np.nan_to_num(S)
print(f"rows {N} | metrics {M} | token feats {F} | static {S.shape[1]}", flush=True)

# ---------- model ----------
class TMAE(nn.Module):
    def __init__(s):
        super().__init__()
        s.inp = nn.Linear(2 * F, D); s.sin = nn.Linear(2 * S.shape[1], D)
        s.pos = nn.Parameter(torch.randn(5, D) * 0.02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 4 * D, 0.1, batch_first=True, norm_first=True), 3)
        s.out = nn.Linear(D, F); s.sout = nn.Linear(D, S.shape[1])
    def forward(s, x, m, xs, ms):          # m/ms: 1 = visible
        tok = s.inp(torch.cat([x * m, m], -1)); st = s.sin(torch.cat([xs * ms, ms], -1))[:, None]
        h = s.enc(torch.cat([tok, st], 1) + s.pos)
        return s.out(h[:, :4]), s.sout(h[:, 4])

def make_mask(B, rng):
    m = (rng.random((B, 4, F)) > 0.25)
    per = rng.random(B) < 0.3; m[per, rng.integers(0, 4, per.sum())] = False                       # whole period
    col = rng.random(B) < 0.4; m[col, :, rng.integers(0, F, col.sum())] = False                     # whole metric across time
    ms = rng.random((B, S.shape[1])) > 0.25
    return m, ms

tx, to = torch.tensor(seq, dtype=torch.float32), torch.tensor(obs, dtype=torch.float32)
ts, tso = torch.tensor(S, dtype=torch.float32), torch.tensor(Sobs, dtype=torch.float32)
net = TMAE(); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * math.ceil(N / 512))
rng = np.random.default_rng(SEED); w = torch.ones(N); hub = nn.HuberLoss(reduction="none", delta=1.0)
t0 = time.time()
for ep in range(EPOCHS):
    net.train(); perm = torch.randperm(N); row_loss = torch.zeros(N)
    for i in range(0, N, 512):
        b = perm[i:i + 512]; m, ms = make_mask(len(b), rng)
        m, ms = torch.tensor(m, dtype=torch.float32) * to[b], torch.tensor(ms, dtype=torch.float32) * tso[b]
        pr, ps = net(tx[b], m, ts[b], ms)
        tgt = (1 - m) * to[b]; tgs = (1 - ms) * tso[b]
        l = (hub(pr, tx[b]) * tgt).sum((1, 2)) / (tgt.sum((1, 2)) + 1) + 0.3 * (hub(ps, ts[b]) * tgs).sum(1) / (tgs.sum(1) + 1)
        loss = (l * w[b]).sum() / w[b].sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        row_loss[b] = l.detach()
    if ep >= 5:                                                                                      # robust: trim likely anomalies
        cut = torch.quantile(row_loss, 1 - TRIM); w = torch.where(row_loss > cut, torch.tensor(0.05), torch.tensor(1.0))
    if ep % 10 == 0 or ep == EPOCHS - 1: print(f"ep {ep}: loss {row_loss.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)

# ---------- scoring: deterministic masking passes ----------
net.eval(); err = np.zeros((N, 4, F)); cnt = np.zeros((N, 4, F)); serr = np.zeros((N, S.shape[1]))
passes = [("period", p) for p in range(4)] + [("metric", f) for f in range(F)]
with torch.no_grad():
    for kind, k in passes:
        for i in range(0, N, 4096):
            sl = slice(i, i + 4096); B = tx[sl].shape[0]; m = torch.ones(B, 4, F)
            if kind == "period": m[:, k] = 0
            else: m[:, :, k] = 0
            m = m * to[sl]; pr, ps = net(tx[sl], m, ts[sl], tso[sl])
            e = ((pr - tx[sl]) ** 2 * (1 - m) * to[sl]).numpy(); err[sl] += e; cnt[sl] += ((1 - m) * to[sl]).numpy()
    # static: mask each static column
    for j in range(S.shape[1]):
        for i in range(0, N, 4096):
            sl = slice(i, i + 4096); ms = tso[sl].clone(); ms[:, j] = 0
            _, ps = net(tx[sl], to[sl], ts[sl], ms); serr[sl, j] = ((ps[:, j] - ts[sl, j]) ** 2 * tso[sl, j]).numpy()
E = err / np.maximum(cnt, 1)                                                                          # (N,4,F)
cells = np.concatenate([E.reshape(N, -1), serr], 1)
med = np.median(cells, 0); mad = 1.4826 * np.median(np.abs(cells - med), 0) + 1e-6
Z = (cells - med) / mad
score = np.sort(Z, 1)[:, -8:].mean(1)                                                                # top-8 cells
os.makedirs(OUT, exist_ok=True)
pd.DataFrame({"tmae": score}).to_csv(f"{OUT}/task5_tmae_scores.csv", index=False)

# ---------- proxy check vs known groups ----------
if os.path.exists(GROUPS):
    Gp = pd.read_csv(GROUPS); r = pd.Series(score).rank(ascending=False).values
    print("\nrecall of known groups inside TMAE top-N (no labels used in training):")
    for n in (1000, 1811, 2500):
        print(f"  top {n}: " + " | ".join(f"{c} {int(((r <= n) & (Gp[c] == 1)).sum())}/{int(Gp[c].sum())}" for c in ["rule", "bot", "lowact", "lowcomm", "lowstab"]))
    U = (Gp[["rule", "bot", "lowact", "lowcomm", "lowstab"]].sum(1) > 0).values
    from sklearn.metrics import roc_auc_score
    print(f"  AUC vs union-1811 proxy: {roc_auc_score(U, score):.4f} | vs rule-827: {roc_auc_score(Gp.rule, score):.4f}")
print("wrote", OUT, f"| {time.time()-t0:.0f}s")
