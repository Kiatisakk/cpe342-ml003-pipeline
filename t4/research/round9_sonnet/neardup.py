import pandas as pd, numpy as np
from PIL import Image
R='cpe342-kaggle-main/04_image_classification/'
tr=pd.read_csv(R+'train.csv').sample(8000,random_state=1); te=pd.read_csv(R+'test.csv').sample(2000,random_state=1)
def f(d,i): return np.asarray(Image.open(f'{R}{d}/{i}.jpg').convert('L').resize((16,8)),dtype=np.float32).ravel()
A=np.stack([f('train',i) for i in tr.id]); B=np.stack([f('test',i) for i in te.id])
A-=A.mean(1,keepdims=True); B-=B.mean(1,keepdims=True)
D=((B**2).sum(1)[:,None]+(A**2).sum(1)[None]-2*B@A.T)
m=D.min(1); print(np.percentile(m,[0,1,5,50]).round(1))
# within-train nn for reference
DA=((A**2).sum(1)[:,None]+(A**2).sum(1)[None]-2*A@A.T); np.fill_diagonal(DA,1e9); print(np.percentile(DA.min(1),[0,1,5,50]).round(1))
j=D.argmin(1); lab=tr['class'].values[j]
print('n test with nn dist<50:',(m<50).sum())
