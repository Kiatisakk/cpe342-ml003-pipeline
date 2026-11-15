from ref import *
from svr import prep
import torch,torch.nn as nn
torch.set_num_threads(2)
def fit(Xa,ya,Xb,Xt):
    torch.manual_seed(0)
    pa,pb,pt=[torch.tensor(q,dtype=torch.float32) for q in prep(Xa,Xb,Xt)]
    S=10000.;yt=torch.tensor(ya/S,dtype=torch.float32)
    m=nn.Sequential(nn.Linear(pa.shape[1],256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
    E=14;opt=torch.optim.AdamW(m.parameters(),2e-3,weight_decay=1e-4);n=len(pa)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,4e-3,total_steps=E*((n+511)//512))
    for e in range(E):
        perm=torch.randperm(n)
        for i in range(0,n,512):
            ix=perm[i:i+512];opt.zero_grad();l=(m(pa[ix]).squeeze(1)-yt[ix]).abs().mean();l.backward();opt.step();sch.step()
    f=lambda P:np.maximum(m(P).squeeze(1).detach().numpy(),0)*S
    return f(pb),f(pt)
if __name__=='__main__':
    o,t=run('nn_mlp_l1',fit);np.save('out/nn_mlp_l1_oof.npy',o);np.save('out/nn_mlp_l1_test.npy',t)
