"""Replace the task5 column of an existing full submission with a T5 candidate vector (no Kaggle call).

Usage (from repo root):
    python t123/research/t5/t5_swap_into_submission.py submissions/sub_v6_t1gbmblend_t2nb_t3blend_t4ens5f_ovr_t5union.csv \
        t123/research/t5/t5_candidates/t5_union_plus_acckda_p50.csv  submissions/sub_v7_t5acckda_p50.csv
Checks: same ids in the same order, 25,889 rows; writes floats, comma, LF, other columns untouched.
"""
import sys, pandas as pd
base, cand, out = sys.argv[1:4]
S = pd.read_csv(base); C = pd.read_csv(cand)
assert len(S) == len(C) == 25889 and (S["id"].values == C["id"].values).all(), "id mismatch"
old = (S["task5"] > 0.5).sum()
S["task5"] = C["pred"].astype(float)
S.to_csv(out, index=False, lineterminator="\n")
print(f"{out}: task5 flagged {old} -> {int(C['pred'].sum())}; other columns unchanged")
