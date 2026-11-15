from common import *
def prep(df):
    X=feats(df,SIG,True)
    for c in ['level_progression_speed','fr_age','rep_age','age_lo','age_hi','lv_lo','lv_hi','r']:
        X[c]=np.log1p(X[c].clip(lower=0))
    X['age_hi']=X.age_hi.clip(upper=8);X['age_lo']=X.age_lo.clip(upper=8)
    M=df[SIG].isna().astype(float).add_suffix('_na')
    return pd.concat([X,M],axis=1)
