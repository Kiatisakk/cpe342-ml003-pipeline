import pandas as pd,numpy as np
pd.set_option('display.width',250,'display.max_rows',500)
d=pd.read_csv('data/test.csv')
print(d.player_id.is_unique)
x=d.drop(columns='player_id')
s=x.describe(percentiles=[.001,.01,.5,.99,.999]).T[['count','min','0.1%','1%','50%','99%','99.9%','max','std']]
s['nuniq']=x.nunique()
print(s.round(3).to_string())
