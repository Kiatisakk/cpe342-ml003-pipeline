import numpy as np,pandas as pd,os,glob
from PIL import Image
R='cpe342-kaggle-main/04_image_classification/'; rng=np.random.default_rng(0)
tr=pd.read_csv(R+'train.csv'); te=pd.read_csv(R+'test.csv')
def st(i,d):
    im=Image.open(f'{R}{d}/{i}.jpg').convert('RGB'); a=np.asarray(im,float)
    hf=np.abs(np.diff(a,axis=1)).mean(); return dict(w=im.size[0],h=im.size[1],mean=a.mean(),std=a.std(),hf=hf,black=(a.sum(2)<30).mean(),kb=os.path.getsize(f'{R}{d}/{i}.jpg')/1024)
a=pd.DataFrame([st(i,'train') for i in rng.choice(tr.id,1500,replace=False)]); b=pd.DataFrame([st(i,'test') for i in rng.choice(te.id,1500,replace=False)])
print(pd.concat({'train':a.describe().loc[['mean','25%','50%','75%']].T,'test':b.describe().loc[['mean','25%','50%','75%']].T},axis=1).round(2).to_string())
print('overlap ids',len(set(tr.id)&set(te.id)))
