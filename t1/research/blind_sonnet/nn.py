from prep import *
import torch,torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
torch.set_num_threads(2)
Xa=prep(tr).values;Xb=prep(te).values
def fp(A,ya,Bs):
    pp=make_pipeline(SimpleImputer(strategy='median'),QuantileTransformer(n_quantiles=200,output_distribution='normal',subsample=50000,random_state=0)).fit(A)
    t=lambda x:torch.tensor(np.nan_to_num(pp.transform(x)),dtype=torch.float32)
    A_=t(A);y_=torch.tensor(ya,dtype=torch.float32)
    torch.manual_seed(0)
    m=nn.Sequential(nn.Linear(A_.shape[1],128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    opt=torch.optim.AdamW(m.parameters(),2e-3,weight_decay=1e-3);E=12
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,4e-3,total_steps=E*((len(A_)+511)//512))
    for e in range(E):
        m.train();perm=torch.randperm(len(A_))
        for i in range(0,len(A_),512):
            j=perm[i:i+512];opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(m(A_[j]).squeeze(1),y_[j]).backward();opt.step();sch.step()
    m.eval()
    with torch.no_grad():return [torch.sigmoid(m(t(x)).squeeze(1)).numpy() for x in Bs]
run('nn_mlp',fp,Xa,Xb,inner=2,save=True)
