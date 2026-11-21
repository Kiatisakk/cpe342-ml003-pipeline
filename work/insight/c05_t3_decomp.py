# T3: where the absolute error of the best OOF comes from (zero/pay uncertainty, whale spikes, multiplicative noise).
import pandas as pd, numpy as np
tr=pd.read_csv("cpe342-kaggle-main/03_regression/train.csv"); o=pd.read_csv("t3/research/best_oof.csv")
y=o.y.values; p=o.pred.values; assert np.allclose(y,tr.spending_30d.values)
e=np.abs(y-p); tot=e.sum(); print("OOF score %.5f  MAE/mean %.4f"%(1/(1+tot/y.sum()),tot/y.sum()))
res=pd.read_csv("work/insight/c02_resid.csv").set_index("player_id").res; r=tr.player_id.map(res).values
seg=tr.segment.fillna(-1).values
cat=np.select([(y==0)&(p==0),(y==0)&(p>0),(y>0)&(p==0),(y>0)&(r>0.8)],["true zero, pred 0","zero but pred>0","payer but pred 0","whale spike (y redrawn)"],"payer, multiplicative noise")
d=pd.DataFrame({"cat":cat,"e":e,"seg":seg,"y":y})
t=d.groupby("cat").agg(rows=("e","size"),err_share=("e",lambda v:v.sum()/tot),mean_err=("e","mean")).sort_values("err_share",ascending=False); print(t.round(4).to_string())
print("\nerror share by segment x source"); print((d.pivot_table(index="cat",columns="seg",values="e",aggfunc="sum")/tot).round(4).fillna(0).to_string())
print("\nshare of total y by segment:",(d.groupby("seg").y.sum()/y.sum()).round(4).to_dict(),"| error share by segment:",(d.groupby("seg").e.sum()/tot).round(4).to_dict())
# oracle bounds: (1) spikes removed from the metric, (2) zeros known
m=cat!="whale spike (y redrawn)"; print("\nscore if spikes were excluded: %.5f"%(1/(1+e[m].sum()/y[m].sum())))
e2=np.where(y==0,0,e); print("score if every zero were known (payers unchanged): %.5f"%(1/(1+e2.sum()/y.sum())))
pay=(y>0)&(p>0)&m; print("payers predicted >0, non-spike: median |log(y/pred)| %.4f ; mean |y-p|/y %.4f"%(np.median(np.abs(np.log(y[pay]/p[pay]))),np.mean(e[pay]/y[pay])))
