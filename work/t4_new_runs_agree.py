# Test-set agreement matrix of T4 runs (argmax of TTA-mean probas, OLD test order). Checks row order of new runs
# (a wrong order gives ~20-30% agreement) and diversity. Also prints predicted class shares.
import sys, numpy as np
sys.argv = ["x", "n", "v3", "clean"]
src = open("work/t4_build_subset.py", encoding="utf-8").read().split("SH = float")[0]
exec(src)                                  # defines FT, T, V, C
keys = ["v3", "cnxtBf", "cnxtLf", "cnxtLfk", "eva02b", "cnxtLv3", "cnxtLv3k", "eva02Lf"]
P = {k: T(*FT[k]) for k in keys}
for k in keys: assert P[k].shape == P["v3"].shape, (k, P[k].shape)
A = {k: P[k].argmax(1) for k in keys}
print("agreement (test, argmax)")
print(" " * 9 + "".join(f"{k:>9}" for k in keys))
for a in keys: print(f"{a:>9}" + "".join(f"{(A[a] == A[b]).mean():9.3f}" for b in keys))
print("\nclass share", C)
for k in keys: print(f"{k:>9}", np.round(np.bincount(A[k], minlength=5) / len(A[k]), 3), "mean max-p", round(float(P[k].max(1).mean()), 3))
