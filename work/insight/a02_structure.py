# Structural artefacts: label vs id / row order, train-test duplicate values, decimal precision by class.
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
D="cpe342-kaggle-main/"
T={"t1":("01_binary_classification","is_cheater"),"t2":("02_multiclass_classification","segment"),"t3":("03_regression","spending_30d")}
for k,(d,tg) in T.items():
    tr=pd.read_csv(f"{D}{d}/train.csv"); te=pd.read_csv(f"{D}{d}/test.csv")
    print(f"\n===== {k}  train {tr.shape} label NaN {tr[tg].isna().mean():.4f}")
    idn=tr.player_id.str[1:].astype(int).values; row=np.arange(len(tr)); y=tr[tg].values; ok=~np.isnan(y)
    yb = (y>0).astype(float) if k=="t3" else y
    print(" spearman(label,id) %.4f  spearman(label,row) %.4f"%(spearmanr(y[ok],idn[ok])[0],spearmanr(y[ok],row[ok])[0]))
    # lag-1 autocorrelation of label in row order and in id order
    for nm,o in (("row",row),("id",np.argsort(idn))):
        z=yb[o]; z=z[~np.isnan(z)]; z=(z>np.median(z)).astype(float) if k=="t3" else z
        print(f"  lag1 autocorr in {nm} order %.4f; block-mean sd (100 blocks) %.4f vs binomial %.4f"%(np.corrcoef(z[:-1],z[1:])[0,1], np.std([b.mean() for b in np.array_split(z,100)]), z.std()/np.sqrt(len(z)/100)))
    if k=="t1": # unlabeled rows: where?
        u=~ok; print("  unlabeled: spearman(unl,row) %.4f (unl,id) %.4f"%(spearmanr(u,row)[0],spearmanr(u,idn)[0]))
        num=tr.drop(columns=["player_id",tg]).select_dtypes("number")
        aucs={c:roc_auc_score(u[num[c].notna()],num[c].dropna()) for c in num}; s=pd.Series(aucs); print("  unlabeled-vs-feature AUC range %.3f..%.3f; NaN-rate diff max %.4f"%(s.min(),s.max(),(num[u].isna().mean()-num[~u].isna().mean()).abs().max()))
    # test ids vs train ids interleave?
    tid=te.player_id.str[1:].astype(int).values
    print("  KS-like: mean id train %.0f test %.0f; test row order vs id spearman %.4f"%(idn.mean(),tid.mean(),spearmanr(np.arange(len(te)),tid)[0]))
    # exact duplicated values between train and test, per column (continuous cols)
    num=[c for c in te.columns if c!="player_id" and te[c].dtype!=object]
    print("  col: nunique_train | share of test values present in train | max decimals | frac integer")
    for c in num:
        a=tr[c].dropna(); b=te[c].dropna(); nu=a.nunique()
        if nu<200: continue
        sh=b.isin(set(a.values)).mean()
        dec=a.sample(3000,random_state=0).map(lambda v: len(repr(float(v)).split(".")[1]) if "e" not in repr(float(v)) else 99)
        print(f"   {c:34s} {nu:7d} {sh:6.3f}  dec p50 {int(dec.median()):2d} max {int(dec.max()):2d}  int {np.mean(a==a.round()):.3f}")
    # full-row duplicates on the 5 most continuous columns
    cc=sorted(num,key=lambda c:-tr[c].nunique())[:4]
    m=te[cc].dropna().merge(tr[cc+[tg]].dropna(subset=cc),on=cc,how="inner"); print("  test rows matching a train row on",cc,":",len(m))
