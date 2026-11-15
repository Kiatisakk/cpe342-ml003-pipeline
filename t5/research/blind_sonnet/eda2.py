from feat import *
pd.set_option('display.width',250,'display.max_rows',500)
rows=[]
for m in metrics:
    if m in('login_lat','login_lon'):continue
    X=d[[f'{m}_{i}' for i in range(1,5)]].values
    # period-wise corr between periods: are 1,2,3,4 correlated within-row?
    c=pd.DataFrame(X).corr().values
    base=np.nanmean(X[:,:2],1)
    for p in (2,3,4):
        pass
    dz3=(X[:,2]-base); dz4=(X[:,3]-base)
    r=[m,c[0,1],c[0,2],c[0,3]]
    for dz in (dz3,dz4):
        s=mad(dz); z=dz/s
        r+= [np.nansum(np.abs(z)>6),np.nansum(np.abs(z)>4)]
    rows.append(r)
print(pd.DataFrame(rows,columns=['m','c12','c13','c14','n3>6','n3>4','n4>6','n4>4']).round(3).to_string())
