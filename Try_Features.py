import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import  seaborn as sns
from datetime import date
import statsmodels.api as sm

def cond(i):
    if i % 7 == 5:
        return 1
    elif i % 7 == 4:
        return 1
    return 0


media = pd.read_csv("data/mediacompany.csv")
media = media.drop("Unnamed: 7", axis=1)
#print(media.head())

media['Date'] = pd.to_datetime(media['Date'])
#print(media.head())

d0 = np.datetime64(date(2017, 2, 20))
d1 = media["Date"].values
media["day"] = d1 - d0
#print(media.head())

media["day"] = media["day"].astype(str)
media["day"] = media["day"].map(lambda x: x[0:2])
media["day"] = media["day"].astype(int)

media.plot(x="day", y="Views_show")
plt.show()

area = np.pi * 3
plt.scatter(media.day, media.Views_show, s=area, c="red", alpha=0.5)
plt.title("Scatter plot ")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()
par2 = host.twinx()

host.set_xlabel("Day")
host.set_ylabel("View_Show")
par1.set_ylabel("Ad_impression")

color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.5)
color3 = plt.cm.viridis(.9)

p1, = host.plot(media.day, media.Views_show, color=color1, label="View_Show")
p2, = par1.plot(media.day, media.Ad_impression, color=color2, label="Ad_impression")
lns = [p1, p2]
host.legend(handles=lns, loc='best')
# right, left, top, bottom
par2.spines['right'].set_position(('outward', 60))
# no x-ticks
par2.xaxis.set_ticks([])
# Sometimes handy, same for xaxis
#par2.yaxis.set_ticks_position('right')
host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
plt.show()
plt.savefig("pyplot_multiple_y-axis.png", bbox_inches='tight')

media["weekday"] = (media["day"]+3) % 7
media.weekday.replace(0, 7, inplace=True)
media["weekday"] = media["weekday"].astype(int)
#print(media.head())


x = media[["Visitors", "weekday"]]
y = media["Views_show"]
x = sm.add_constant(x)
lm_ols_1 = sm.OLS(y, x).fit()
print(lm_ols_1.summary())

media["weekend"] = [cond(i) for i in media["day"]]
#print(media.head())
x = media[["Visitors", "weekend"]]
x = sm.add_constant(x)
lm_ols_2 = sm.OLS(y, x).fit()
print(lm_ols_2.summary())


x = media[["Visitors", "weekend", "Character_A"]]
x = sm.add_constant(x)
lm_ols_3 = sm.OLS(y, x).fit()
print(lm_ols_3.summary())

media["Lag_Views"] = np.roll(media["Views_show"], 1)
media.Lag_Views.replace(media["Lag_Views"].iloc[0], 0, inplace=True)
#print(media.head())
x = media[["Visitors", "Lag_Views", "Character_A", "weekend"]]
x = sm.add_constant(x)
lm_ols_4 = sm.OLS(y, x).fit()
print(lm_ols_4.summary())


plt.figure(figsize=(20, 10))
sns.heatmap(media.corr(), annot=True)
plt.show()

x = media[["weekend", "Character_A", "Views_platform"]]
x = sm.add_constant(x)
lm_ols_5 = sm.OLS(y, x).fit()
print(lm_ols_5.summary())

x = media[["weekend", "Character_A", "Visitors"]]
x = sm.add_constant(x)
lm_ols_6 = sm.OLS(y, x).fit()
print(lm_ols_6.summary())


x = media[["weekend", "Character_A", "Visitors", "Ad_impression"]]
x = sm.add_constant(x)
lm_ols_7 = sm.OLS(y, x).fit()
print(lm_ols_7.summary())


x = media[["weekend", "Character_A" , "Ad_impression"]]
x = sm.add_constant(x)
lm_ols_8 = sm.OLS(y, x).fit()
print(lm_ols_8.summary())

media["ad_impression_million"] = media["Ad_impression"]/1000000

x = media[["weekend", "Character_A", "ad_impression_million", "Cricket_match_india"]]
x = sm.add_constant(x)
lm_ols_9 = sm.OLS(y, x).fit()
print(lm_ols_9.summary())

x = media[["weekend", "Character_A", "ad_impression_million"]]
x = sm.add_constant(x)
lm_ols_10 = sm.OLS(y, x).fit()
print(lm_ols_10.summary())



