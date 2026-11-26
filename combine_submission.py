"""Combine predictions from Tasks 1-5 into a single Kaggle submission CSV.

Usage:
    python combine_submission.py

Expects each task notebook (`task1/task1.ipynb` .. `task5/task5.ipynb`) to have
generated `task<N>/task<N>_predictions.csv` with columns `id,pred`, and writes
`submission.csv` (25,889 rows: `id,task1,task2,task3,task4,task5`).
"""

import glob
import os
import pandas as pd


def find_sample_submission() -> str:
    candidates = [
        os.path.join("cpe342-kaggle-main", "sample_submission.csv"),
        "sample_submission.csv",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    hits = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(
        "sample_submission.csv not found. Place the Kaggle dataset under cpe342-kaggle-main/."
    )


def main() -> None:
    sample = pd.read_csv(find_sample_submission())
    expected_ids = sample["id"].values
    out = pd.DataFrame({"id": expected_ids})

    task_specs = [
        ("task1", os.path.join("task1", "task1_predictions.csv"), "int"),
        ("task2", os.path.join("task2", "task2_predictions.csv"), "int"),
        ("task3", os.path.join("task3", "task3_predictions.csv"), "float"),
        ("task4", os.path.join("task4", "task4_predictions.csv"), "int"),
        ("task5", os.path.join("task5", "task5_predictions.csv"), "int"),
    ]

    for col, path, kind in task_specs:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {path}. Run {col}/{col}.ipynb first to generate it."
            )
        df = pd.read_csv(path)
        if len(df) != len(expected_ids) or not (df["id"].values == expected_ids).all():
            raise ValueError(f"ID alignment mismatch in {path}.")
        if df["pred"].isna().any():
            raise ValueError(f"NaN predictions found in {path}.")
        out[col] = df["pred"].astype(int) if kind == "int" else df["pred"].astype(float).clip(lower=0.0)

    out_path = "submission.csv"
    out.to_csv(out_path, index=False, lineterminator="\n")
    print(f"Saved {out_path} | shape={out.shape}")
    print(f"  task1 positive rate : {out['task1'].mean():.4f}")
    print(f"  task2 class dist    : {out['task2'].value_counts(normalize=True).sort_index().round(4).to_dict()}")
    print(f"  task3 zero / mean   : {(out['task3'] == 0).mean():.4f} / {out['task3'].mean():.2f}")
    print(f"  task4 class dist    : {out['task4'].value_counts(normalize=True).sort_index().round(4).to_dict()}")
    print(f"  task5 flagged count : {int(out['task5'].sum()):,} ({out['task5'].mean():.4f})")


if __name__ == "__main__":
    main()
