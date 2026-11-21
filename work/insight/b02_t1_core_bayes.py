# T1: inside overlap cells (uniform features carry no information there) compare the final GBM's OOF with parametric discriminants on the
# skill features. If LDA/QDA matches the GBM, the GBM already sits at the Bayes rate of the Gaussian part.
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
tr=pd.read_csv("cpe342-kaggle-main/01_binary_classification/train.csv"); lab=tr[tr.is_cheater.notna()].reset_index(drop=True)
oof=np.load("t1/research/best_oof_avg3.npy"); assert len(oof)==len(lab); y=lab.is_cheater.values.astype(int)
print("final model overall AUC %.4f"%roc_auc_score(y,oof))
SK=["kill_death_ratio","headshot_percentage","win_rate","accuracy_score","kill_consistency","reaction_time_ms","aiming_smoothness","spray_control_score","game_sense_score"]
a,f,r,l,c=lab.account_age_days,lab.friend_network_size,lab.reports_received,lab.level,lab.crosshair_placement
cells={"age 200-1000, friends 100-249, reports 5-19, level 28-96, crosshair .82-.94":(a.between(200,1000)&f.between(100,249)&r.between(5,19)&l.between(28,96)&c.between(.82,.94)),
       "age 15-110, friends 10-49, reports 10-19, level 28-96, crosshair .82-.94":(a.between(15,110)&f.between(10,49)&r.between(10,19)&l.between(28,96)&c.between(.82,.94)),
       "age 120-180, friends 45-89, reports 5-19, level 28-96, crosshair .82-.94":(a.between(120,180)&f.between(45,89)&r.between(5,19)&l.between(28,96)&c.between(.82,.94))}
cv=StratifiedKFold(5,shuffle=True,random_state=0)
for nm,m in cells.items():
    m=m&lab[SK].notna().all(axis=1); X=lab.loc[m,SK].values; yy=y[m]; X2=np.column_stack([X,lab.loc[m,["friend_network_size","reports_received","account_age_days","level","crosshair_placement","level_progression_speed"]].values])
    print(f"\n{nm}: rows {m.sum()} cheater rate {yy.mean():.3f}")
    print("   final GBM OOF (all features)        AUC %.4f"%roc_auc_score(yy,oof[m]))
    for mn,est,XX in (("LDA skill 9",LDA(),X),("QDA skill 9",QDA(reg_param=.01),X),("logistic skill 9",make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000)),X),
                      ("logistic quadratic skill 9",make_pipeline(StandardScaler(),PolynomialFeatures(2),LogisticRegression(C=.05,max_iter=3000)),X),
                      ("logistic skill 9 + 6 cell feats",make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000)),X2)):
        p=cross_val_predict(est,XX,yy,cv=cv,method="predict_proba")[:,1]; print(f"   {mn:35s} AUC %.4f | rank-avg with GBM %.4f"%(roc_auc_score(yy,p),roc_auc_score(yy,pd.Series(p).rank().values+pd.Series(oof[m]).rank().values)))
