# T5 OLD file (id leak): are anomaly types laid out in contiguous player_id blocks? If so the block sizes give the generator's exact type counts.
import pandas as pd, numpy as np, glob
P=[1,2,3,4]; STATIC=["behavior_stability","account_age_days"]
BOX={"bot":{"ip_hash_entropy":(0,.2),"kda_std":(.05,.15),"login_time_variance":(.1,.5),"performance_stability":(.95,.99),"play_pattern_consistency":(.95,.99),"session_duration_std":(1,5),"device_count":(1,1)},
     "lowact":{"activity_level":(.1,.3),"purchase_amount_mean":(300,500),"skill_rating":(2200,2500)},
     "lowcomm":{"communication_frequency":(0,.1),"friend_adds":(15,25),"guild_changes":(2,3)},
     "lowstab":{"behavior_stability":(.2,.4),"account_age_days":(800,1500),"purchase_frequency":(3,6)},
     "acckda":{"performance_stability":(.85,.95),"kda_std":(.7,1.0),"accuracy_std":(9,12)}}
X=pd.read_csv(glob.glob("data_v1_backup/**/05_anomaly_detection/test.csv",recursive=True)[0]); cols=lambda b:[b] if b in STATIC else [f"{b}_{p}" for p in P]
idn=X.player_id.str[1:].astype(int).values
typ=np.array(["-"]*len(X),dtype=object)
for g,bx in BOX.items():
    ok=np.ones(len(X),bool)
    for b,(lo,hi) in bx.items():
        m=X[cols(b)].values.astype(float); ok&=(((m>=lo)&(m<=hi))|np.isnan(m)).all(1)
    typ[ok&(typ=="-")]=g
blk=idn>=23819; print("block rows",blk.sum(),"| type counts inside block:",pd.Series(typ[blk]).value_counts().to_dict()); print("type counts outside block:",pd.Series(typ[~blk]).value_counts().to_dict())
o=np.argsort(idn); s=pd.DataFrame({"id":idn[o],"typ":typ[o]}); s=s[s.id>=23819].reset_index(drop=True)
# run-length encode the type sequence by id
runs=[]; st=0
for i in range(1,len(s)+1):
    if i==len(s) or s.typ[i]!=s.typ[st]: runs.append((s.id[st],s.id[i-1],s.typ[st],i-st)); st=i
print("number of runs:",len(runs)); print("first 15 runs:",runs[:15])
# type composition per 100-id window
s["w"]=(s.id-23819)//207; print(pd.crosstab(s.w,s.typ).to_string())
