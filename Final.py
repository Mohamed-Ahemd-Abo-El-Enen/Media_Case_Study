import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import  seaborn as sns
from datetime import date
import statsmodels.api as sm
from sklearn.metrics import  mean_squared_error, r2_score


media = pd.read_csv("data/mediacompany.csv")
media = media.drop("Unnamed: 7", axis=1)

media['Date'] = pd.to_datetime(media['Date'])
d0 = np.datetime64(date(2017, 2, 20))
d1 = media["Date"].values
media["day"] = d1 - d0
media["day"] = media["day"].astype(str)
media["day"] = media["day"].map(lambda x: x[0:2])
media["day"] = media["day"].astype(int)

media["Lag_Views"] = np.roll(media["Views_show"], 1)
media.Lag_Views.replace(media["Lag_Views"].iloc[0], 0, inplace=True)

media["ad_impression_million"] = media["Ad_impression"]/1000000

plt.figure(figsize=(20, 10))
sns.heatmap(media.corr(), annot=True)
plt.show()

x = media.drop(["Views_show", "Date"], axis=1)
plt.figure(figsize=(20, 10))
sns.heatmap(x.corr(), annot=True)
plt.show()

y = media["Views_show"]
x = sm.add_constant(x)

lm_ols = sm.OLS(y, x).fit()
print(lm_ols.summary())

predict_views = lm_ols.predict(x)
mse = mean_squared_error(media.Views_show, predict_views)
r_squared = r2_score(media.Views_show, predict_views)
print("MSE : ", mse)
print("R Squared Value : ", r_squared)


c = [i for i in range(0, len(y))]
fig = plt.figure()
plt.plot(c, media["Views_show"], color="green", linewidth=2.5, linestyle='-')
plt.plot(c, predict_views, color="blue", linewidth=2.5, linestyle='-')
fig.suptitle("Actual and predicted", fontsize=20)
plt.xlabel("index", fontsize=18)
plt.ylabel("views", fontsize=16)
plt.show()


e = [i for i in range(0, len(y))]
fig = plt.figure()
plt.plot(e, (media["Views_show"] - predict_views), color="red", linewidth=2.5, linestyle='-')
fig.suptitle("Error Term", fontsize=20)
plt.xlabel("index", fontsize=18)
plt.ylabel("view_show - predicted_view", fontsize=16)
plt.show()