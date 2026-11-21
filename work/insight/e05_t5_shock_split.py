# T5 NEW: assign the 827 shock rows to takeover / selling / VPN with NaN fallbacks; expect 310 / 310 / 207 if the generator layout is unchanged.
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
for nm,f,g in (("old","data_v1_backup/05_anomaly_detection/test.csv","submissions/probes/probes_t5/t5_groups.csv"),("new","cpe342-kaggle-main/05_anomaly_detection/test.csv","work/t5v2/out/t5_groups.csv")):
    X=pd.read_csv(f); G=pd.read_csv(g); R=X[G.rule.values==1]
    r=lambda b:(R[f"{b}_4"]/R[[f"{b}_{p}" for p in (1,2,3)]].mean(axis=1)).replace(np.inf,99)
    s=pd.DataFrame({"cur":r("currency_transfers"),"loc":r("location_changes"),"frm":r("friend_removes"),"pf":r("purchase_frequency"),"pa":r("purchase_amount_mean"),"con":r("play_pattern_consistency"),"kda":r("kda_mean"),"acc":r("accuracy_mean"),"skt":(R.skill_trajectory_4-R[[f"skill_trajectory_{p}" for p in (1,2,3)]].mean(axis=1)).abs()})
    tk=((s.con<0.7).astype(int)+(s.cur>3)); se=((s.kda>1.3).astype(int)+(s.frm>3)+(s.acc>1.3)); vp=((s.pa>1.6).astype(int)+(s.pf>3))
    sc=pd.DataFrame({"takeover":tk,"selling":se,"vpn":vp}); t=sc.idxmax(axis=1).where(sc.max(axis=1)>0,"?"); tie=(sc.eq(sc.max(axis=1),axis=0).sum(axis=1)>1)&(sc.max(axis=1)>0)
    print(nm,"shock rows",len(R),t.value_counts().to_dict(),"| ties",int(tie.sum()))
    if (t=="?").sum(): print(s[t=="?"].round(2).to_string())
    if nm=="new":
        P=[1,2,3,4]; base=sorted({c[:-2] for c in X.columns if c[-2:] in ("_1","_2","_3","_4")})
        allnan=pd.DataFrame({b:X[[f"{b}_{p}" for p in P]].isna().all(axis=1) for b in base}); print("rows with some metric NaN in all 4 periods:",int(allnan.any(axis=1).sum()),"(expected %.1f under MCAR)"%(len(base)*len(X)*0.0488**4)); print("  by metric:",allnan.sum()[allnan.sum()>0].to_dict())
