"""Task 5 - refine the new acc/kda/perf cluster: model variants + held-out-feature calibration.

Usage (from repo root, after t5_pipeline.py):
    python t5/research/t5_cluster_refine.py [test.csv] [out/t5_pipeline_scores.csv]
Writes t5/research/out/acckda_posteriors.csv (posterior of every variant + ensemble).

Variants (EM: Gaussian cluster vs product-of-marginals background, rows outside the union):
  M3 = accuracy_std, kda_std, performance_stability (4-period means)
  M4 = M3 + performance_stability~vol   (level-adjusted log std over periods; cluster has ~half the noise)
  M6 = M4 + accuracy_std~vol + behavior_stability
Calibration check (model-free): in each posterior band of a variant, the mean of a feature the variant did
NOT use is a mixture of the "core" (posterior > 0.97) and "normal" (posterior < 0.01) means, so
frac = (mean_band - mean_normal) / (mean_core - mean_normal) estimates the true share of cluster rows in that
band. A calibrated posterior gives frac ~ mean posterior of the band.
"""
import sys, numpy as np, pandas as pd, warnings
from scipy.stats import multivariate_normal
warnings.filterwarnings("ignore")
SRC = sys.argv[1] if len(sys.argv) > 1 else "cpe342-kaggle-main/05_anomaly_detection/test.csv"
SC = sys.argv[2] if len(sys.argv) > 2 else "t5/research/out/t5_pipeline_scores.csv"
X = pd.read_csv(SRC); S = pd.read_csv(SC); P = [1, 2, 3, 4]
un = S[["shock"] + [c for c in S.columns if c[:1] == "G" and c[1:].isdigit()]].max(axis=1).values == 1
rest = ~un
mean = lambda b: X[[f"{b}_{p}" for p in P]].astype(float).mean(axis=1).values
F = pd.DataFrame({"accuracy_std": mean("accuracy_std"), "kda_std": mean("kda_std"),
                  "performance_stability": mean("performance_stability"), "behavior_stability": X.behavior_stability.values})
def vol(b, ref):
    m = X[[f"{b}_{p}" for p in P]].values.astype(float); mu = np.nanmean(m, 1); l = np.log(np.nanstd(m, 1) + 1e-9)
    ok = ref & ~np.isnan(m).any(1); co = np.polyfit(mu[ok], l[ok], 3)
    v = l - np.polyval(co, mu); v[np.isnan(m).any(1)] = np.nan; return v

def em(cols, init):
    Y0 = F.loc[rest, cols]; Y0 = Y0.fillna(Y0.median()); Y = ((Y0 - Y0.mean()) / Y0.std()).values
    bg = np.ones(len(Y))
    for j in range(Y.shape[1]):
        h, e = np.histogram(Y[:, j], bins=200); w = e[1] - e[0]
        d = np.convolve(h, np.ones(5) / 5, "same") / (len(Y) * w)
        bg *= np.maximum(d[np.clip(((Y[:, j] - e[0]) / w).astype(int), 0, 199)], 1e-6)
    r = init[rest] + 1e-6
    for _ in range(200):
        pi = r.mean(); mu = (r[:, None] * Y).sum(0) / r.sum(); dd = Y - mu
        C = (r[:, None, None] * np.einsum("ij,ik->ijk", dd, dd)).sum(0) / r.sum()
        c = multivariate_normal(mu, C).pdf(Y); r = pi * c / (pi * c + (1 - pi) * bg)
    out = np.zeros(len(F)); out[rest] = r; return out

init = S["post_C0"].values
F["performance_stability~vol"] = vol("performance_stability", rest & (init < 0.01))
F["accuracy_std~vol"] = vol("accuracy_std", rest & (init < 0.01))
VAR = {"M3": ["accuracy_std", "kda_std", "performance_stability"]}
VAR["M4"] = VAR["M3"] + ["performance_stability~vol"]
VAR["M6"] = VAR["M4"] + ["accuracy_std~vol", "behavior_stability"]
post = {k: em(c, init) for k, c in VAR.items()}
post["ENS"] = (post["M3"] + post["M4"] + post["M6"]) / 3

print("variant  n_eff  p>0.1  p>0.5  p>0.9  ambiguous(0.1-0.9)  expTP@0.1")
for k, p in post.items():
    print(f"{k:6s} {p.sum():6.1f} {int((p > .1).sum()):6d} {int((p > .5).sum()):6d} {int((p > .9).sum()):6d} {int(((p > .1) & (p < .9)).sum()):10d} {p[p > .1].sum():14.1f}")

print("\nHeld-out calibration (frac = share of true cluster rows in the band, from a feature the model did not use)")
HELD = {"M3": ["performance_stability~vol", "accuracy_std~vol", "behavior_stability"],
        "M4": ["accuracy_std~vol", "behavior_stability"], "M6": []}
for k in ["M3", "M4"]:
    p = post[k]; core = rest & (p > 0.97); norm = rest & (p < 0.01)
    print(f"{k}: core n={core.sum()}")
    for lo, hi in [(0.02, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 0.97)]:
        b = rest & (p > lo) & (p <= hi); txt = []
        for h in HELD[k]:
            v = F[h].values; den = np.nanmean(v[core]) - np.nanmean(v[norm])
            txt.append(f"{h} {(np.nanmean(v[b]) - np.nanmean(v[norm])) / den:5.2f}±{np.nanstd(v[b]) / np.sqrt(b.sum()) / abs(den):.2f}")
        print(f"   band ({lo},{hi}] n={b.sum():4d} mean posterior {p[b].mean():.2f} | " + "  ".join(txt))

out = pd.DataFrame({"id": [f"ANS{i + 1:05d}" for i in range(len(X))], "union": un.astype(int)})
for k, p in post.items(): out[f"post_{k}"] = p.round(4)
out.to_csv("t5/research/out/acckda_posteriors.csv", index=False)
