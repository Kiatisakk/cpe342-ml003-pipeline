# Build T4 string-label prediction files (goal C). Run from repo root:  python t123/research/t4/t4_make_label_csv.py
# Source = v3_5f fold-ensemble test probabilities, plain argmax (prior-shift correction was NOT adopted, see REPORT.md).
#   t4_test_str.csv       id,pred   labels exactly as in train.csv  (7-11k, fiveN, freefried, pube, roblock)
#   t4_test_str_711K.csv  id,pred   same, but 7-11k written as 7-11K (spelling on slide page 30)
# id = ANS00001.. in test.csv row order (same order as sample_submission / every test.csv).
import numpy as np, pandas as pd
D = "t123/preds_final/t4/preds"; OUT = "t123/research/t4"
tr = pd.read_csv("cpe342-kaggle-main/04_image_classification/train.csv")
te = pd.read_csv("cpe342-kaggle-main/04_image_classification/test.csv")
C = sorted(tr["class"].unique()); assert C == ["7-11k", "fiveN", "freefried", "pube", "roblock"], C
P = np.load(f"{D}/task4_v3_5f_test_proba.npy"); assert P.shape == (len(te), 5)
old = pd.read_csv(f"{D}/task4_v3_5f.csv")
assert (old["id_img"].values == te["id"].values).all(), "row order mismatch with test.csv"
ids = old["id"].values; assert ids[0] == "ANS00001" and len(set(ids)) == len(te) == 25889
pred = np.array(C)[P.argmax(1)]
assert (pred == old["pred_str"].values).all()
pd.DataFrame({"id": ids, "pred": pred}).to_csv(f"{OUT}/t4_test_str.csv", index=False, lineterminator="\n")
pd.DataFrame({"id": ids, "pred": np.where(pred == "7-11k", "7-11K", pred)}).to_csv(f"{OUT}/t4_test_str_711K.csv", index=False, lineterminator="\n")
print(pd.Series(pred).value_counts().to_dict()); print("wrote t4_test_str.csv, t4_test_str_711K.csv")
