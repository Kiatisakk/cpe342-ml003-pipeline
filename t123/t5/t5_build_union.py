"""Task 5 — rebuild the rule + stationary-cluster anomaly set used in sub_v6 (union, ~1811 rows).
Usage: python t5_build_union.py <path/to/05_anomaly_detection/test.csv> <out.csv>
Output columns: id, rule, bot, lowact, lowcomm, lowstab, t5 (union)
"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
X = pd.read_csv(sys.argv[1]); P = [1, 2, 3, 4]

# --- geo: login location in periods 3/4 vs home (mean of periods 1-2) ---
lat = np.stack([X[f"login_lat_{p}"] for p in P], 1).astype(float)
lon = np.stack([X[f"login_lon_{p}"] for p in P], 1).astype(float)
bad_lat = (np.abs(lat) > 90).any(1)
lt = np.where(np.abs(lat) > 90, np.nan, lat)
hl, hn = np.nanmean(lt[:, :2], 1), np.nanmean(lon[:, :2], 1)
jump = np.nan_to_num(np.nanmax(np.hypot(lt[:, 2:] - hl[:, None], (lon[:, 2:] - hn[:, None] + 180) % 360 - 180), 1))

# --- shock: change of periods 3/4 vs mean(1,2), robust z (median/MAD) per metric; need >=2 metrics with z>6 ---
bases = sorted({c[:-2] for c in X.columns if c[-2:] in ("_1", "_2", "_3", "_4")} - {"login_lat", "login_lon"})
Z = []
for b in bases:
    m = X[[f"{b}_{p}" for p in P]].astype(float).values
    ch = np.nanmax(np.abs(m[:, 2:] - np.nanmean(m[:, :2], 1, keepdims=True)), 1)
    med = np.nanmedian(ch); mad = 1.4826 * np.nanmedian(np.abs(ch - med)) + 1e-9
    Z.append((ch - med) / mad)
z2 = np.sort(np.nan_to_num(np.stack(Z, 1)), 1)[:, -2]

rule = (jump > 5) | bad_lat | (z2 > 6)
mean4 = lambda b: X[[f"{b}_{p}" for p in P]].astype(float).mean(axis=1).values
bot = mean4("login_time_variance") < 1.0
lowact = mean4("activity_level") < 0.34
lowcomm = mean4("communication_frequency") < 0.17
lowstab = X["behavior_stability"].values < 0.55
IDC = X.columns[0]
out = pd.DataFrame({"id": X[IDC], "rule": rule, "bot": bot, "lowact": lowact, "lowcomm": lowcomm, "lowstab": lowstab}).astype({c: int for c in ["rule", "bot", "lowact", "lowcomm", "lowstab"]})
out["t5"] = out[["rule", "bot", "lowact", "lowcomm", "lowstab"]].max(axis=1)
out.to_csv(sys.argv[2], index=False)
print(out[["rule", "bot", "lowact", "lowcomm", "lowstab", "t5"]].sum().to_dict())
