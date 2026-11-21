# T1 generator shape: per class x age sub-population, marginal moments and bounds of each signal feature; within-group correlations.
import pandas as pd, numpy as np, sys
from scipy.stats import skew, kurtosis
sys.path.insert(0,"t1/research"); pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
tr=pd.read_csv("cpe342-kaggle-main/01_binary_classification/train.csv"); lab=tr[tr.is_cheater.notna()].copy()
S=["kill_death_ratio","headshot_percentage","win_rate","accuracy_score","kill_consistency","reaction_time_ms","account_age_days","level","level_progression_speed","friend_network_size","reports_received","aiming_smoothness","spray_control_score","game_sense_score","crosshair_placement"]
lab["grp"]=pd.cut(lab.account_age_days,[-np.inf,15,110,120,180,200,1000,np.inf],labels=["<15","15-110","110-120","120-180","180-200","200-1000",">1000"])
print(pd.crosstab(lab.grp,lab.is_cheater,dropna=False))
for c in S:
    print(f"\n--- {c}")
    r=[]
    for (g,y),d in lab.groupby(["grp","is_cheater"],observed=True):
        v=d[c].dropna().values
        if len(v)<200: continue
        r.append((g,int(y),len(v),v.min(),np.quantile(v,.01),np.median(v),v.mean(),np.quantile(v,.99),v.max(),v.std(),skew(v),kurtosis(v),np.mean(v==v.min()),np.mean(v==v.max())))
    print(pd.DataFrame(r,columns="grp y n min q01 med mean q99 max sd skew kurt atmin atmax".split()).round(3).to_string(index=False))
