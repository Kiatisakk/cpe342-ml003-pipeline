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
    "t5_union_plus_acckda_p50": union | (post > 0.5),       # recommended first probe
    "t5_union_plus_acckda_p10": union | (post > 0.1),       # F3-optimal threshold (marginal precision ~ F3/10)
    "t5_union_plus_acckda_p90": union | (post > 0.9),       # most conservative
}
rows = []
for name, m in cands.items():
    pd.DataFrame({"id": ANS, "pred": m.astype(int)}).to_csv(f"{OUT}/{name}.csv", index=False, lineterminator="\n")
    add = m & ~union
    rows.append((name, int(m.sum()), int(add.sum()), round(float(post[add].sum()), 1)))
print(pd.DataFrame(rows, columns=["file", "n_flagged", "n_added", "expected_TP_added"]).to_string(index=False))
