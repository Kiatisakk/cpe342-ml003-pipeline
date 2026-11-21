# T2: do numeric features depend on the categorical levels (engagement_level / loyalty_tier / vip_tier) WITHIN a class?
# If yes, features are not conditionally independent given the class and a hierarchical latent model beats class-NB.
import pandas as pd, numpy as np
pd.set_option("display.width",250)
tr=pd.read_csv("cpe342-kaggle-main/02_multiclass_classification/train.csv")
E=["play_frequency","avg_session_duration","total_playtime_hours","login_streak","days_since_last_login","friend_count","chat_activity_score","tournament_entries","speed_of_progression","peak_concurrent_hours"]
S=["total_spending_thb","avg_monthly_spending","spending_frequency","rare_items_count","vip_tier","item_type_preference_cosmetic"]
print("mean of engagement features by engagement_level within class (z-scored per class)")
for c in E[:5]+S[:3]:
    z=(tr[c]-tr.groupby("segment")[c].transform("mean"))/tr.groupby("segment")[c].transform("std")
    print(c); print(z.groupby([tr.segment,tr.engagement_level.fillna("NaN")]).mean().unstack().round(3).to_string())
print("\nmean of spending features by loyalty_tier within class (z-scored per class)")
for c in S+E[:2]:
    z=(tr[c]-tr.groupby("segment")[c].transform("mean"))/tr.groupby("segment")[c].transform("std")
    print(c); print(z.groupby([tr.segment,tr.loyalty_tier.fillna("NaN")]).mean().unstack().round(3).to_string())
print("\nwithin-class correlation matrix (pooled, class-demeaned) of key features")
K=E+S; Z=tr[K]-tr.groupby("segment")[K].transform("mean"); print(Z.corr().round(3).to_string())
print("\nengagement_level x loyalty_tier within class 0 (independence?)"); d=tr[tr.segment==0]; print(pd.crosstab(d.engagement_level,d.loyalty_tier,normalize="index").round(3))
