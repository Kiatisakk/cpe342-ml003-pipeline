import warnings;warnings.filterwarnings('ignore')
from feat import *
pd.set_option('display.width',250,'display.max_rows',500)
skip={'login_lat','login_lon','ip_hash_entropy','device_count','refund_count','guild_changes'}
Z4={};Z3={}
for m in metrics:
    if m in skip: continue
    X=d[[f'{m}_{i}' for i in range(1,5)]].values
    b12=np.nanmean(X[:,:2],1); b13=np.nanmean(X[:,:3],1)
    for P,key,b,Zd in ((3,2,b12,Z3),(4,3,b13,Z4)):
        dz=X[:,key]-b
        Zd[m]=dz/mad(dz)
Z4=pd.DataFrame(Z4);Z3=pd.DataFrame(Z3)
big4=(Z4.abs()>6)
print('rows with >=k big p4 shifts:',[(k,int((big4.sum(1)>=k).sum())) for k in range(1,12)])
big3=(Z3.abs()>6)
print('rows with >=k big p3 shifts:',[(k,int((big3.sum(1)>=k).sum())) for k in range(1,8)])
# pattern signatures for rows with >=2 big p4
sel=big4.sum(1)>=2
sg=(np.sign(Z4[sel]).fillna(0)*big4[sel]).astype(int)
print(sg.apply(lambda r:"".join("-+."[0] if v<0 else ("+" if v>0 else ".") for v in r),axis=1).value_counts().head(20))
print(list(Z4.columns))
Z4.to_pickle('out/Z4.pkl');Z3.to_pickle('out/Z3.pkl')
