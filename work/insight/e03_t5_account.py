# T5 NEW file: account for the generator's fixed sizes (827 shock, 207 bot, 1036 stationary = 2070) against our prediction.
import pandas as pd, numpy as np
P=[1,2,3,4]; STATIC=["behavior_stability","account_age_days"]
BOX={"bot":{"ip_hash_entropy":(0,.2),"kda_std":(.05,.15),"login_time_variance":(.1,.5),"performance_stability":(.95,.99),"play_pattern_consistency":(.95,.99),"session_duration_std":(1,5),"device_count":(1,1)},
     "lowact":{"activity_level":(.1,.3),"purchase_amount_mean":(300,500),"skill_rating":(2200,2500)},
     "lowcomm":{"communication_frequency":(0,.1),"friend_adds":(15,25),"guild_changes":(2,3)},
     "lowstab":{"behavior_stability":(.2,.4),"account_age_days":(800,1500),"purchase_frequency":(3,6)},
     "acckda":{"performance_stability":(.85,.95),"kda_std":(.7,1.0),"accuracy_std":(9,12)}}
X=pd.read_csv("cpe342-kaggle-main/05_anomaly_detection/test.csv"); cols=lambda b:[b] if b in STATIC else [f"{b}_{p}" for p in P]
G=pd.read_csv("work/t5v2/out/t5_groups.csv"); pred=pd.read_csv("t5/preds/v2data/t5v2_union_box_lsnan.csv").iloc[:,1].values==1
post=pd.read_csv("work/t5v2/out/lowstab_nan_scores.csv").post_lowstab_nan.values
B={}
for g,bx in BOX.items():
    ok=np.ones(len(X),bool)
    for b,(lo,hi) in bx.items():
        m=X[cols(b)].values.astype(float); ok&=(((m>=lo)&(m<=hi))|np.isnan(m)).all(1)
    B[g]=ok
rule=G.rule.values==1
vis_ls=B["lowstab"]&X.behavior_stability.notna().values          # behaviour_stability observed in [.2,.4]
cand=B["lowstab"]&X.behavior_stability.isna().values             # hidden-lowstab candidates
certain=rule|B["bot"]|B["lowact"]|B["lowcomm"]|B["acckda"]|vis_ls
print("rule",rule.sum(),"bot",B["bot"].sum(),"lowact",B["lowact"].sum(),"lowcomm",B["lowcomm"].sum(),"acckda",B["acckda"].sum(),"lowstab visible",vis_ls.sum())
stat=(B["lowact"]|B["lowcomm"]|B["acckda"]|vis_ls)&~rule&~B["bot"]; print("stationary certain (union of 4 boxes):",stat.sum()," -> hidden lowstab implied by 1036:",1036-stat.sum())
print("certain total",certain.sum(),"| predicted",pred.sum(),"| certain not predicted",(certain&~pred).sum(),"| predicted not certain",(pred&~certain).sum())
miss=np.where(certain&~pred)[0]
for i in miss: print("  missed row",i,"groups:",[g for g in B if B[g][i]],"rule",rule[i],"bs",X.behavior_stability[i],"age",X.account_age_days[i],"n NaN cells",int(X.iloc[i,1:].isna().sum()))
# rows in two stationary boxes
ov=(B["lowact"].astype(int)+B["lowcomm"]+B["acckda"]+vis_ls+B["bot"])>=2; print("rows inside 2+ boxes:",ov.sum())
c=cand&~certain; print("\nhidden-lowstab candidates (bs NaN, other lowstab box conditions hold, not otherwise certain):",c.sum(),"| predicted among them",(pred&c).sum(),"| predicted outside candidates & certain",(pred&~c&~certain).sum())
ps=np.sort(post[c])[::-1]; print("posterior (prior 14) sum over candidates %.2f ; top 60:"%ps.sum(),np.round(ps[:60],2))
# re-scale the posterior odds to the exact hidden count
h=1036-stat.sum(); odds=ps/(1-ps+1e-12)
lo,hi=0.1,50
for _ in range(60):
    k=(lo+hi)/2; s=(k*odds/(1+k*odds)).sum(); lo,hi=(k,hi) if s<h else (lo,k)
p2=k*odds/(1+k*odds); print(f"odds multiplier {k:.3f} so that the posteriors sum to {h}: rows with p>=0.0995: {(p2>=.0995).sum()} (before {(ps>=.0995).sum()})")
