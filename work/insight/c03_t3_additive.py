# T3: is log(y/prev) additive in the features (multiplicative factors)? Shape of each factor; pure-noise floor from exact cells.
import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from scipy.stats import skew, kurtosis
pd.set_option("display.width",250)
tr=pd.read_csv("cpe342-kaggle-main/03_regression/train.csv"); y=tr.spending_30d
F=[c for c in tr.columns if c not in ("player_id","spending_30d")]
pos=tr[(y>0)&tr.prev_month_spending.notna()&tr.segment.notna()].reset_index(drop=True); pos["lr"]=np.log(pos.spending_30d/pos.prev_month_spending)
res=pd.read_csv("work/insight/c02_resid.csv").res.values; spike=res>0.8; print("rows",len(pos),"spikes dropped",spike.sum())
pos=pos[~spike].reset_index(drop=True)
def cv(depth,leaves,n=800,obj="l2"):
    o=np.zeros(len(pos))
    for a,b in KFold(5,shuffle=True,random_state=0).split(pos):
        o[b]=lgb.LGBMRegressor(n_estimators=n,learning_rate=.05,max_depth=depth,num_leaves=leaves,min_child_samples=50,objective=obj,verbose=-1).fit(pos.loc[a,F],pos.lr[a]).predict(pos.loc[b,F])
    return o
for nm,(d,l) in {"stumps (additive)":(1,2),"depth 2":(2,4),"depth 3":(3,8),"leaves 31":(-1,31)}.items():
    o=cv(d,l); r=pos.lr-o; print(f"{nm:18s} resid sd {r.std():.4f}  mean|r| {np.abs(r).mean():.4f}  kurt {kurtosis(r):.3f}")
# factor shapes: mean of lr by level of the discrete drivers, within segment 1 (cleanest mid segment)
for c in ["is_premium_member","vip_status","games_played","cross_game_activity","guild_membership","owns_limited_edition","seasonal_spending_pattern","payment_methods_used","platform","primary_game","tournament_participation"]:
    t=pos.groupby(["segment",c]).lr.mean().unstack(0).round(3); 
    if len(t)<=12: print(f"\n{c} (mean log r by level x segment; NaN level omitted)"); print(t.to_string())
for c in ["purchase_frequency","days_since_last_purchase","event_participation_rate","total_transactions","discount_rate_used"]:
    q=pd.qcut(pos[c],12,duplicates="drop"); print(f"\n{c} deciles: mean log r (all segments, segment-demeaned)"); d=pos.lr-pos.groupby("segment").lr.transform("mean")
    print(d.groupby(q).mean().round(3).to_string())
