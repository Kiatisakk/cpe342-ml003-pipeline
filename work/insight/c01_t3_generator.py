# T3 generator shape: zero rate and y/prev ratio per segment; which raw columns move the ratio; ratio shape (uniform? lognormal?).
import pandas as pd, numpy as np
from scipy.stats import skew, kurtosis, spearmanr
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
tr=pd.read_csv("cpe342-kaggle-main/03_regression/train.csv"); y=tr.spending_30d
print("y: zeros %.4f  min pos %.4f  max %.1f  decimals sample"%((y==0).mean(), y[y>0].min(), y.max()), y[y>0].head(5).tolist())
print("segment NaN %.4f; prev NaN %.4f; prev==0 %.4f; hist==0 %.4f"%(tr.segment.isna().mean(),tr.prev_month_spending.isna().mean(),(tr.prev_month_spending==0).mean(),(tr.historical_spending==0).mean()))
g=tr.groupby(tr.segment.fillna(-1))
print(g.apply(lambda d: pd.Series({"n":len(d),"zero":(d.spending_30d==0).mean(),"y_mean":d.spending_30d.mean(),"y_med_pos":d.spending_30d[d.spending_30d>0].median(),"prev_med":d.prev_month_spending.median(),"prev0":(d.prev_month_spending==0).mean(),"hist_med":d.historical_spending.median()})).round(3))
pos=tr[(y>0)&(tr.prev_month_spending>0)].copy(); pos["r"]=pos.spending_30d/pos.prev_month_spending; pos["lr"]=np.log(pos.r)
print("\nratio r=y/prev among payers, per segment")
print(pos.groupby(pos.segment.fillna(-1)).r.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).round(3))
print(pos.groupby(pos.segment.fillna(-1)).lr.agg(["mean","std",skew,kurtosis]).round(3))
print("\nspearman of log r with each column (payers, segment-demeaned)")
pos["lrd"]=pos.lr-pos.groupby(pos.segment.fillna(-1)).lr.transform("mean")
num=[c for c in tr.columns if c not in ("player_id","spending_30d") and tr[c].dtype!=object]
print(pd.Series({c:spearmanr(pos[c],pos.lrd,nan_policy="omit")[0] for c in num}).sort_values().round(3).to_string())
print("\nobject cols:",{c:tr[c].nunique() for c in tr.columns if tr[c].dtype==object and c!="player_id"})
print("\nzero rate vs each column (spearman, segment-demeaned zero flag)")
z=(y==0).astype(float); zd=z-z.groupby(tr.segment.fillna(-1)).transform("mean")
print(pd.Series({c:spearmanr(tr[c],zd,nan_policy="omit")[0] for c in num}).sort_values().round(3).to_string())
