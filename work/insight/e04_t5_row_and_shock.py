# T5 NEW: (1) inspect the lowact-box row our gap rule missed; (2) split the 827 shock rows into subtypes to test the 310/310/207 layout.
import pandas as pd, numpy as np
X=pd.read_csv("cpe342-kaggle-main/05_anomaly_detection/test.csv"); G=pd.read_csv("work/t5v2/out/t5_groups.csv"); P=[1,2,3,4]
i=25766; print("row",i,X.player_id[i])
for b,(lo,hi) in {"activity_level":(.1,.3),"purchase_amount_mean":(300,500),"skill_rating":(2200,2500)}.items():
    v=X.loc[i,[f"{b}_{p}" for p in P]].values.astype(float); allv=X[[f"{b}_{p}" for p in P]].values.ravel(); allv=allv[~np.isnan(allv)]
    print(f"  {b:22s} {np.round(v,3)} box [{lo},{hi}] | share of all cells inside the box {np.mean((allv>=lo)&(allv<=hi)):.4f}")
la=G.lowact.values==1; print("  gap-rule lowact rows: mean activity range",X.loc[la,[f"activity_level_{p}" for p in P]].mean(axis=1).min().round(3),X.loc[la,[f"activity_level_{p}" for p in P]].mean(axis=1).max().round(3), "| this row's mean",np.nanmean(X.loc[i,[f"activity_level_{p}" for p in P]].values.astype(float)).round(3))
# chance that a normal row with these cells observed sits inside all three boxes
obs={b:X.loc[i,[f"{b}_{p}" for p in P]].notna().values for b in ("activity_level","purchase_amount_mean","skill_rating")}
norm=~(G.iloc[:,1:].max(axis=1).values==1); pr=1
for b,(lo,hi) in {"activity_level":(.1,.3),"purchase_amount_mean":(300,500),"skill_rating":(2200,2500)}.items():
    m=X.loc[norm,[f"{b}_{p}" for p in P]].values[:,obs[b]]; ok=(((m>=lo)&(m<=hi))|np.isnan(m)).all(1); print(f"  normal rows inside the {b} box on the observed periods {obs[b].sum()}: {ok.mean():.5f}"); pr*=ok.mean()
print("  -> chance for a normal row %.2e (x 23.8k rows = %.4f expected)"%(pr,pr*norm.sum()))
# (2) shock subtypes
R=X[G.rule.values==1]; r=lambda b:(R[f"{b}_4"]/R[[f"{b}_{p}" for p in (1,2,3)]].mean(axis=1))
sig=pd.DataFrame({"currency":r("currency_transfers"),"loc":r("location_changes"),"friend_rm":r("friend_removes"),"purch_freq":r("purchase_frequency"),"purch_amt":r("purchase_amount_mean"),"consist":r("play_pattern_consistency"),"kda":r("kda_mean")})
print("\nshock rows",len(R)); print(sig.describe(percentiles=[.1,.5,.9]).round(2).to_string())
t=np.where(sig.consist<0.7,"takeover",np.where(sig.kda>1.3,"selling",np.where(sig.purch_amt>1.6,"vpn","?")))
print(pd.Series(t).value_counts().to_dict())
q=sig[t=="?"]; print("unassigned:",len(q)); print(q.round(2).head(12).to_string())
