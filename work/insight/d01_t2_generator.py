# T2 generator shape: per-class marginal moments and bounds of every feature; categorical tables.
import pandas as pd, numpy as np
from scipy.stats import skew, kurtosis
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
tr=pd.read_csv("cpe342-kaggle-main/02_multiclass_classification/train.csv")
print(tr.segment.value_counts(normalize=True).sort_index().round(4).to_dict())
num=[c for c in tr.columns if c not in ("player_id","segment") and pd.api.types.is_numeric_dtype(tr[c])]; cat=[c for c in tr.columns if not pd.api.types.is_numeric_dtype(tr[c]) and c!="player_id"]
rows=[]
for c in num:
    for s,d in tr.groupby("segment"):
        v=d[c].dropna().values
        rows.append((c,s,tr[c].nunique(),round(d[c].isna().mean(),3),v.min(),np.quantile(v,.01),np.median(v),v.mean(),np.quantile(v,.99),v.max(),v.std(),skew(v),kurtosis(v),np.mean(v==v.min()),np.mean(v==v.max())))
R=pd.DataFrame(rows,columns="feature seg nuniq nan min q01 med mean q99 max sd skew kurt atmin atmax".split())
print(R.round(3).to_string(index=False))
for c in cat:
    print(f"\n{c}: NaN {tr[c].isna().mean():.3f}"); print(pd.crosstab(tr[c].fillna("NaN"),tr.segment,normalize="columns").round(3).to_string())
