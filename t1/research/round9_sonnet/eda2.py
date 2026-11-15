import pandas as pd,numpy as np
D='cpe342-kaggle-main/01_binary_classification/'
a=pd.read_csv(D+'train.csv');l=a[a.is_cheater.notna()]
fe=[c for c in a.columns if c not in('player_id','is_cheater')]
nm=l[fe].isna().sum(1);print(l.groupby(nm).is_cheater.agg(['mean','size']).T.round(3).to_string())
for c in fe:
    v=l[c].dropna();vc=v.value_counts();top=vc.head(3)
    if vc.iloc[0]>5:
        print(c,'min',v.min(),'max',v.max(),'top',[(k,n,round(l.loc[l[c]==k,'is_cheater'].mean(),3)) for k,n in top.items()])
