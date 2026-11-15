import pandas as pd,numpy as np,time
from sklearn.model_selection import KFold
D='data/';tr=pd.read_csv(D+'train.csv');te=pd.read_csv(D+'test.csv')
y=tr.spending_30d.values
def score(y,p): return 1/(1+np.mean(np.abs(y-p))/np.mean(np.abs(y)))
folds=list(KFold(5,shuffle=True,random_state=42).split(tr))
base=[c for c in te.columns if c!='player_id']
def feats(df):
    X=df[base].copy()
    tx,av,hs,pm=X.total_transactions,X.avg_transaction_value,X.historical_spending,X.prev_month_spending
    # exact fills
    X['historical_spending']=hs.fillna(tx*av)
    X['total_transactions']=tx.fillna(X.historical_spending/av)
    X['avg_transaction_value']=av.fillna(X.historical_spending/X.total_transactions)
    hs=X.historical_spending
    X['purchases_on_discount']=X.purchases_on_discount.fillna(X.total_transactions*X.discount_rate_used)
    X['lpm']=np.log1p(pm);X['lhs']=np.log1p(hs)
    X['pm_hs']=pm/(hs+1);X['pm_tx']=pm/(X.total_transactions+1)
    X['hs_age']=hs/(X.account_age_days+1)
    return X.replace([np.inf,-np.inf],np.nan)
