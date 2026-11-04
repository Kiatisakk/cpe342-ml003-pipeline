"""Task 5 - write submission-ready T5 vectors (id, pred 0/1, test.csv row order).

Usage (from repo root, after t5_pipeline.py):
    python t123/research/t5/t5_make_candidates.py [t123/research/t5/out/t5_pipeline_scores.csv] [t123/research/t5/t5_candidates]
"""
import sys, os, pandas as pd, numpy as np
SC = sys.argv[1] if len(sys.argv) > 1 else "t123/research/t5/out/t5_pipeline_scores.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "t123/research/t5/t5_candidates"
os.makedirs(OUT, exist_ok=True)
R = pd.read_csv(SC)
assert len(R) == 25889
ANS = [f"ANS{i + 1:05d}" for i in range(len(R))]   # submission id = test.csv row order (scores file keeps player_id)
gcols = [c for c in R.columns if c[:1] == "G" and c[1:].isdigit()]
union = (R[["shock"] + gcols].max(axis=1) == 1)
post = R["post_C0"]
cands = {
    "t5_union_1811":            union,                      # reference = sub_v6 T5 (pipeline reproduction)
    "t5_union_plus_acckda_box": union | (R["C0"] == 1),     # RECOMMENDED: exact per-period box (244 rows)
    "t5_union_plus_acckda_box_lsnan": None,                # box + lowstab members hidden by NaN behavior_stability (low confidence)
    "t5_union_plus_acckda_p50": union | (post > 0.5),       # Gaussian-EM posterior variants (superseded by box)
    "t5_union_plus_acckda_p10": union | (post > 0.1),
    "t5_union_plus_acckda_p90": union | (post > 0.9),
}
LS = "t123/research/t5/out/lowstab_nan_scores.csv"   # from t5_box_analysis.py
if os.path.exists(LS):
    cands["t5_union_plus_acckda_box_lsnan"] = cands["t5_union_plus_acckda_box"] | (pd.read_csv(LS)["post_lowstab_nan"].values > 0.1)
else:
    cands.pop("t5_union_plus_acckda_box_lsnan")
rows = []
for name, m in cands.items():
    pd.DataFrame({"id": ANS, "pred": m.astype(int)}).to_csv(f"{OUT}/{name}.csv", index=False, lineterminator="\n")
    add = m & ~union
    rows.append((name, int(m.sum()), int(add.sum()), int((add & (R["C0"] == 1)).sum())))
print(pd.DataFrame(rows, columns=["file", "n_flagged", "n_added", "added_in_box"]).to_string(index=False))
