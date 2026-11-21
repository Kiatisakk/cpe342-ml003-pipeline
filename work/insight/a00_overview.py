import pandas as pd, numpy as np
D="cpe342-kaggle-main/"
T={"t1":"01_binary_classification","t2":"02_multiclass_classification","t3":"03_regression","t5":"05_anomaly_detection"}
for k,d in T.items():
    for s in ("train","test"):
        try: df=pd.read_csv(f"{D}{d}/{s}.csv")
        except FileNotFoundError: continue
        print(k,s,df.shape); print("  cols:",list(df.columns)[:60])
        print("  head ids:",df.iloc[:3,0].tolist(), "dtype0",df.dtypes.iloc[0])
