from common import *
from svm import make,prep
import torch,torch.nn as nn
torch.set_num_threads(2)
A,B=make(False)
def fp(Xa,ya,Xs):
    Za,Zs=prep(Xa,Xs);torch.manual_seed(0)
    m=nn.Sequential(nn.Linear(Za.shape[1],128),nn.SiLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.SiLU(),nn.Dropout(0.2),nn.Linear(64,4))
    opt=torch.optim.AdamW(m.parameters(),2e-3,weight_decay=1e-3);E=12
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,4e-3,total_steps=E*((len(Za)+511)//512))
    X=torch.tensor(Za,dtype=torch.float32);Y=torch.tensor(ya)
    for e in range(E):
        m.train();p=torch.randperm(len(X))
        for i in range(0,len(X),512):
            j=p[i:i+512];opt.zero_grad();nn.functional.cross_entropy(m(X[j]),Y[j]).backward();opt.step();sch.step()
    m.eval()
    with torch.no_grad(): return [torch.softmax(m(torch.tensor(Z,dtype=torch.float32)),1).numpy() for Z in Zs]
cv(fp,'mlp_torch_A',A,B,save=True)
