from common import *
import lightgbm as lgb,os
names=[n for n in ['lgb_l1','hurdle_lgb_quantile','knn_wmedian','extratrees_ratio','nn_mlp_l1','svr_nys_0.02_600','lgb_ratio_wl1'] if os.path.exists(f'out/{n}_oof.npy')]
O=np.column_stack([np.load(f'out/{n}_oof.npy') for n in names]);T=np.column_stack([np.load(f'out/{n}_test.npy') for n in names])
print(names)
# nested: LAD linear stacker via LGBM-free: scipy minimize abs error on weights fit on 4 folds, evaluated on 5th
from scipy.optimize import minimize
def fitw(A,yy):
    k=A.shape[1];f=lambda w:np.mean(np.abs(yy-A@w))
    return minimize(f,np.ones(k)/k,method='Powell',options={'maxiter':3000}).x
s=[];oof=np.zeros(len(y))
for a,b in folds:
    w=fitw(O[a],y[a]);oof[b]=np.maximum(O[b]@w,0);s.append(score(y[b],oof[b]))
print('stack_lad_nested',np.mean(s),np.std(s))
w=fitw(O,y);print(dict(zip(names,w.round(3))))
np.save('out/stack_lad_oof.npy',oof);np.save('out/stack_lad_test.npy',np.maximum(T@w,0))
# simple mean of top
for combo in [['lgb_l1','hurdle_lgb_quantile'],['lgb_l1','hurdle_lgb_quantile','nn_mlp_l1']]:
    if all(c in names for c in combo):
        p=O[:,[names.index(c) for c in combo]].mean(1);print(combo,np.mean([score(y[b],p[b]) for a,b in folds]))
