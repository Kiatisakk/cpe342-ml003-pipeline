# T3 noise floor: learning curve of the residual sd of log(y/prev) and a latent-base check (does log(hist/prev) explain the residual?).
import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from scipy.stats import kurtosis
tr=pd.read_csv("cpe342-kaggle-main/03_regression/train.csv"); y=tr.spending_30d
F=[c for c in tr.columns if c not in ("player_id","spending_30d")]
pos=tr[(y>0)&tr.prev_month_spending.notna()&tr.segment.notna()].reset_index(drop=True); pos["lr"]=np.log(pos.spending_30d/pos.prev_month_spending)
res=pd.read_csv("work/insight/c02_resid.csv").res.values; pos=pos[res<=0.8].reset_index(drop=True)
pos["l_hp"]=np.log(pos.historical_spending/pos.prev_month_spending); pos["l_atv"]=np.log(pos.avg_transaction_value/pos.prev_month_spending); pos["l_prev"]=np.log(pos.prev_month_spending)
F2=F+["l_hp","l_atv","l_prev"]
print("hist/prev quantiles per segment:"); print(pos.groupby("segment").l_hp.apply(lambda v:np.exp(np.quantile(v.dropna(),[.01,.25,.5,.75,.99])).round(3)).to_string())
rng=np.random.RandomState(0)
for frac in (.125,.25,.5,1.0):
    o=np.zeros(len(pos))
    for a,b in KFold(5,shuffle=True,random_state=0).split(pos):
        a=rng.choice(a,int(len(a)*frac),replace=False)
        o[b]=lgb.LGBMRegressor(n_estimators=800,learning_rate=.03,max_depth=3,num_leaves=8,min_child_samples=30,verbose=-1).fit(pos.loc[a,F2],pos.lr[a]).predict(pos.loc[b,F2])
    r=pos.lr-o; print(f"train frac {frac:5.3f} ({int(len(pos)*.8*frac):6d} rows): resid sd {r.std():.4f} mean|r| {np.abs(r).mean():.4f} kurt {kurtosis(r):.3f}")
pos["res"]=r
print("corr(res, l_hp) %.4f  corr(res, l_atv) %.4f  corr(res,l_prev) %.4f"%tuple(pos[["l_hp","l_atv","l_prev"]].corrwith(pos.res)))
print("resid by segment: sd, kurt, quantiles"); print(pos.groupby("segment").res.agg(["std",kurtosis,lambda v:np.quantile(v,.001),lambda v:np.quantile(v,.01),lambda v:np.quantile(v,.99),lambda v:np.quantile(v,.999)]).round(4).to_string())
# residual on the ratio scale: y / (prev * exp(fit)); uniform noise would show a flat histogram with hard edges
u=np.exp(pos.res[pos.segment<3]); h,e=np.histogram(u,bins=np.arange(.5,1.62,.04)); print("ratio-scale residual histogram (segments 0-2):"); print(" ".join(f"{a:.2f}:{c}" for a,c in zip(e,h)))
