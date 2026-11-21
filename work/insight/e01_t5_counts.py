# T5: per-type counts with identical box rules on the OLD and NEW test files; NaN rates of the static columns; implied hidden members.
import pandas as pd, numpy as np, glob
P=[1,2,3,4]; STATIC=["behavior_stability","account_age_days"]
BOX={"bot":{"ip_hash_entropy":(0,.2),"kda_std":(.05,.15),"login_time_variance":(.1,.5),"performance_stability":(.95,.99),"play_pattern_consistency":(.95,.99),"session_duration_std":(1,5),"device_count":(1,1)},
     "lowact":{"activity_level":(.1,.3),"purchase_amount_mean":(300,500),"skill_rating":(2200,2500)},
     "lowcomm":{"communication_frequency":(0,.1),"friend_adds":(15,25),"guild_changes":(2,3)},
     "lowstab":{"behavior_stability":(.2,.4),"account_age_days":(800,1500),"purchase_frequency":(3,6)},
     "acckda":{"performance_stability":(.85,.95),"kda_std":(.7,1.0),"accuracy_std":(9,12)}}
files={"old":glob.glob("data_v1_backup/**/05_anomaly_detection/test.csv",recursive=True)[0],"new":"cpe342-kaggle-main/05_anomaly_detection/test.csv"}
for nm,f in files.items():
    X=pd.read_csv(f); cols=lambda b:[b] if b in STATIC else [f"{b}_{p}" for p in P]
    print(f"\n== {nm} {X.shape}  NaN rate behavior_stability {X.behavior_stability.isna().mean():.4f} account_age {X.account_age_days.isna().mean():.4f}; overall cell NaN {X.iloc[:,1:].isna().mean().mean():.4f}")
    tot=np.zeros(len(X),bool)
    for g,bx in BOX.items():
        ok=np.ones(len(X),bool); strict=np.ones(len(X),bool)
        for b,(lo,hi) in bx.items():
            m=X[cols(b)].values.astype(float); ok&=(((m>=lo)&(m<=hi))|np.isnan(m)).all(1); strict&=((m>=lo)&(m<=hi)).all(1)
        # members with the key static metric observed vs missing
        print(f"{g:8s} box rows {ok.sum():4d} (all cells observed {strict.sum():4d}) overlap with earlier boxes {(ok&tot).sum()}"); tot|=ok
    print("any box:",tot.sum())
    if nm=="new":
        ls=np.ones(len(X),bool)
        for b,(lo,hi) in BOX["lowstab"].items():
            m=X[cols(b)].values.astype(float); ls&=(((m>=lo)&(m<=hi))|np.isnan(m)).all(1)
        print("lowstab box by which static is NaN: bs NaN",(ls&X.behavior_stability.isna()).sum(),"| age NaN",(ls&X.account_age_days.isna()).sum(),"| both obs",(ls&X.behavior_stability.notna()&X.account_age_days.notna()).sum())
