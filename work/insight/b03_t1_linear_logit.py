# T1: does the GBM lose efficiency on the weak linear signal of the skill features? Add a cross-fitted logistic "skill logit" (global and per
# age sub-population, mean-impute + missing flags) to fe_v2 and compare LightGBM (screening setup lr 0.05) on the same folds, seeds 42 and 7.
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"t1/research")
import common; common.DATA="cpe342-kaggle-main/01_binary_classification"; common.OUT="work/insight"
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
lab,unl,te,ids=load(); y=lab[TARGET].astype(int).values; ref=pd.concat([lab,unl]); X=fe_v2(lab,ref)
SK=["kill_death_ratio","headshot_percentage","win_rate","accuracy_score","kill_consistency","reaction_time_ms","aiming_smoothness","spray_control_score","game_sense_score"]
Z=lab[SK].copy(); mu=ref[SK].mean(); sd=ref[SK].std(); M=Z.isna().astype(float).add_prefix("na_"); Z=((Z-mu)/sd).fillna(0); D=pd.concat([Z,M],axis=1).values
age=X["age_filled"].values; grp=np.select([age<110,age<200,age<1000],[0,1,2],3); grp=np.where(np.isnan(age),4,grp)
def logit_feat(seed):
    g=np.zeros(len(y)); s=np.zeros(len(y))
    for a,b in folds_for(y,seed):
        m=LogisticRegression(C=1,max_iter=2000).fit(D[a],y[a]); g[b]=m.decision_function(D[b])
        for k in range(5):
            ak=a[grp[a]==k]; bk=b[grp[b]==k]
            if len(np.unique(y[ak]))<2: s[bk]=0; continue
            s[bk]=LogisticRegression(C=1,max_iter=2000).fit(D[ak],y[ak]).decision_function(D[bk])
    return g,s
for seed in (42,7):
    g,s=logit_feat(seed); print(f"seed {seed}: skill logit alone AUC global {roc_auc_score(y,g):.4f} per-group {roc_auc_score(y,s):.4f}",flush=True)
    o0,_=cv_lgb(X,y,seed=seed); r0=report(f"b03 base fe_v2 s{seed}",y,o0,seed,save=False)
    X1=X.copy(); X1["skill_logit"]=g; X1["skill_logit_grp"]=s
    o1,_=cv_lgb(X1,y,seed=seed); r1=report(f"b03 fe_v2 + skill logits s{seed}",y,o1,seed,save=False)
    print(f"   delta AUC {r1['auc']-r0['auc']:+.5f}  F2b nested {r1['f2b_nested']-r0['f2b_nested']:+.5f}  F2m nested {r1['f2m_nested']-r0['f2m_nested']:+.5f}",flush=True)
    np.save(f"work/insight/b03_oof_base_s{seed}.npy",o0); np.save(f"work/insight/b03_oof_logit_s{seed}.npy",o1)
