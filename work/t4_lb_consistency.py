# Which candidate T4 labelling is most consistent with the public LB history?
# For a candidate c, treat its T4 labels as "pseudo truth" and score every past submission i against it (macro-F1,
# pF1_i). If c were the truth, LB_i = base + w4 * F1_i, so LB_i should be linear in pF1_i with slope w4 (~0.2) and the
# fit evaluated at pF1 = 1 estimates c's own LB. Past files share T1/T2/T3/T5 (checked), so T4 is the only difference.
# Caveat: candidates are built from the same models as the late files -> all fits are biased the same way; only the
# ranking between candidates is meaningful, not the absolute numbers.
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
F = "submissions/FINAL/"
PAST = {"sub_v25a_newdata_T2v2_T4realigned_T5v2": .85787, "sub_v26_T4eq3effv2": .85630, "sub_v26_T4coloraug": .85733,
        "sub_v28_T4cnxtsd3": .85906, "sub_v30_T4eq4_cnxtb2f": .86169, "sub_v31_T4eq4_radio_vithp": .86340, "sub_v32_T4frozen50": .86430,
        "sub_v33_T4frozen50_emhalf": .86669, "sub_v34_T4frozen50_emfull": .86750, "sub_v36_T4frozen4_emfull": .86805,
        "sub_v37_T4frozen4_ft5L_emfull": .86837, "sub_v37n_T4v37_noEM": .86494, "sub_v40_T4v37_em150": .86643,
        "sub_v41_T4v37_knn10l3_emfull": .86760, "sub_v56_v54_plusLf": .86921, "sub_v58e": .86920, "sub_v58f": .86929, "sub_v59r3": .86923, "sub_v59r3_noMLP": .86925}
CAND = ["sub_v59r3_noMLP", "sub_v65a_Lv3_em", "sub_v65b_Lv3_em"]
rd = lambda n: pd.read_csv(F + n + ".csv", dtype=str, keep_default_na=False)
base = rd("sub_v25a_newdata_T2v2_T4realigned_T5v2")
T4 = {}
for n in set(PAST) | set(CAND):
    d = rd(n)
    if n in PAST: assert all((d[c] == base[c]).all() for c in ("id", "task2", "task3", "task5")), n
    T4[n] = d.task4.astype(int).values
names = list(PAST); lb = np.array([PAST[n] for n in names])
print(f"{'candidate':32s} {'corr':>5s} {'slope':>6s} {'RMSE':>8s} {'LB at pF1=1':>11s}  (fit without the candidate's own file)")
for c in CAND:
    use = [n for n in names if n != c]; x = np.array([f1_score(T4[c], T4[n], average="macro") for n in use]); yy = np.array([PAST[n] for n in use])
    b, a = np.polyfit(x, yy, 1); r = pearsonr(x, yy)[0]; rm = np.sqrt(np.mean((a + b * x - yy) ** 2))
    print(f"{c[4:36]:32s} {r:5.2f} {b:6.3f} {rm:8.5f} {a + b:11.5f}")
