# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 12:48:04 2022

@author: darvi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# from calendar import day_name
# from collections import deque


#%% Load dataset
df = pd.read_csv("data/df_res_full.csv")


#%% Pie chart function 

Sport_calories_kcal = df.groupby("sport")["calories_kcal"].mean()
Sport_calories_kcal = Sport_calories_kcal.to_frame()
Sport_calories_kcal = Sport_calories_kcal.reset_index()

def pie_chart(data, label):
    
    # Wedge properties
    wp = { 'linewidth' : 1, 'edgecolor' : "black" }
    
    # Creating autocpt arguments
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.1f}%\n({v:1d})'.format(p=pct,v=val)
        return my_autopct
    
    # Creating plot
    fig, ax = plt.subplots(figsize =(36, 32))
    wedges, texts, autotexts = ax.pie(data,
                                      labels = label,
                                      startangle = 90,
                                      shadow = False,
                                      wedgeprops = wp,
                                      autopct = make_autopct(data))
    
    # Adding legend
    ax.legend(wedges, label,
              title = "Sport:",
              loc = "upper right")    
    
    plt.setp(autotexts, size = 16, weight ="bold")
    ax.set_title("Avg. Calories Burned (Kcal) w.r.t Sport", fontsize=36)
    
    # Show plot
    plt.show() 


pie_chart(Sport_calories_kcal["calories_kcal"], Sport_calories_kcal["sport"])

# Comment: This pie chart shows how much Calories(Kcal) on average are burned
# in each Sport.The marks are labeled by Sport and average of Calories Kcal.
# From the pie chart, we can observe that crpss country skiing and both kinds of
# running are the type of sport that burns the most calories on average and they
# contribute to more than a quarter of the chart. On the other hand, it is obvious
# that stretching, cross_training and roller_skating are the type of sport that 
# burns the least calories on average.

#%% Bar plot - WeekDay vs Sport

Sport_duration_s = df.groupby(["sport","start_time_weekday_cat"])["duration_s"].mean()
Sport_duration_s = Sport_duration_s.to_frame()
Sport_duration_s = Sport_duration_s.reset_index()


cats = ['Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
sport_unique = Sport_duration_s["sport"].unique()

dict = {}
for i in sport_unique:
    dict[i] = Sport_duration_s[Sport_duration_s["sport"] == i]
    dict[i] = dict[i].set_index('start_time_weekday_cat').reindex(cats).reset_index()
    dict[i] = dict[i].fillna(0)
    dict[i] = dict[i].to_numpy()
    


plt.subplots(figsize =(20, 23))

for j in sport_unique:
    if j == sport_unique[0]:
        plt.bar(dict[j][:,0].tolist(),dict[j][:,2])
        bottom_ = dict[j][:,2]
    else:    
        plt.bar(dict[j][:,0],dict[j][:,2], bottom = bottom_)
        bottom_ = bottom_ + dict[j][:,2]
        print(bottom_)

 
    
# Adding legend
plt.xlabel("Weekday", fontsize = 23)
plt.ylabel("Sport",fontsize = 23)
plt.legend(sport_unique,
          title = "Sport:",
          loc = "upper center",fontsize = 10)
plt.title("WeekDay vs Sport",fontsize = 30)
plt.show()

# Comment: This stacked bar plot shows the average duration for each weekday.
# Color shows details about Sport. From the chart, we can observe that the
# weekend(saturday and sunday) has the most activity with saturday being the most
# active day of the week and all other week days have on average similar divison 
# of sport activities with Friday being the least active among all comparatively.
# Generally, it seems that running, weight_training, badminton and walking
# are sports contributing most everyday activities including weekends. In contrast,
# there are several activies happing only on some specific days. For instance, 
# beach volley and fitness walking happen only on Sunday or stretch happens only 
# on Tuesday.
#%% Bar plot - Seasons w.r.t sports

Sport_distance_km = df.groupby(["sport","start_time_season_cat"])["distance_km"].mean()
Sport_distance_km = Sport_distance_km.to_frame()
Sport_distance_km = Sport_distance_km.reset_index()


cats = ["AUTUMN", "SPRING", "SUMMER", "WINTER" ]
sport_unique = Sport_distance_km["sport"].unique()

dict = {}
for i in sport_unique:
    dict[i] = Sport_distance_km[Sport_distance_km["sport"] == i]
    dict[i] = dict[i].set_index('start_time_season_cat').reindex(cats).reset_index()
    dict[i] = dict[i].fillna(0)
    dict[i] = dict[i].to_numpy()
    


plt.subplots(figsize =(20, 23))

for j in sport_unique:
    if j == sport_unique[0]:
        plt.bar(dict[j][:,0].tolist(),dict[j][:,2])
        bottom_ = dict[j][:,2]
    else:    
        plt.bar(dict[j][:,0],dict[j][:,2], bottom = bottom_)
        bottom_ = bottom_ + dict[j][:,2]
        print(bottom_)

 
    
# Adding legend
plt.xlabel("Seasons", fontsize = 23)
plt.ylabel("Avs.Distance Km",fontsize = 23)
plt.legend(sport_unique,
          title = "Sport:",
          loc = "upper right",fontsize = 15)
plt.title("Seasons w.r.t sports",fontsize = 30)
plt.show()

# Comment: This stacked bar chart shows the activity for each season. Color shows 
# details about Sport. From the figure, we can observe that the most sports season 
# according to this data is Spring and the least active as can be expected is Winter.
# Auumn and Summer are both similiar in activies. Suprisingly, Summer is less active than 
# Autumn and cycling sport dominates during Spring with significantly high distribution 
# among activities. Fitness walking activity particularly happens only during Spring.


#%% Multivariate Plots

sns.set_style("ticks")
sns.set(font_scale=1.3)
plt.figure(figsize=(30,5))

Sport_duration_s_1 = df.groupby(["sport","start_time_hour"])["duration_s"].mean()
Sport_duration_s_1 = Sport_duration_s_1.to_frame()
Sport_duration_s_1 = Sport_duration_s_1.reset_index()

cats = list(range(2,23))
sport_unique = Sport_duration_s_1["sport"].unique()

dict = {}
for i in sport_unique:
    dict[i] = Sport_duration_s_1[Sport_duration_s_1["sport"] == i]
    dict[i] = dict[i].set_index('start_time_hour').reindex(cats).reset_index()
    dict[i] = dict[i].fillna(0)
    dict[i] = dict[i].to_numpy()

# Final solution
cat = sns.catplot(
    x="start_time_hour", 
    y="duration_s", 
    data=Sport_duration_s_1, 
    height=5,
    aspect=.8,
    kind='point',
    hue='sport', 
    col='sport', 
    col_wrap=6);

cat.fig.subplots_adjust(top=.9)

cat.fig.suptitle("Time vs Sport (24 hr)")

for ax in cat.fig.axes:
    ax.set_xlim(0,22)
    ax.set_xticks(range(2,22,2))
    ax.xaxis.tick_bottom()
    ax.grid(True, axis='both')

cat.set(xlabel="Hourly", ylabel = "Avg of Duration")

# Comment: This plot shows us Average of duration for each time hour broken down
# by Sport. We can observe that walking is the only sport that has activity around 
# the clock. Cycling, swimming, walking, weight_training, and cross country skiing 
# all have similar pattern of activity from 8 AM to 6 PM approximately.
#  And stretching and beach volley is the sport activities  that are done for 
# the specific time and only in the afternoon. However, fitness walking happens
# before noon and last roughly 1.5 hours.


# # Alternative Solution 
# # rel = sns.relplot(x="start_time_hour",
# #                   y="duration_s", 
# #                   data=Sport_duration_s_1, 
# #                   height=5, #default 
# #                   aspect=.8,
# #                   palette='bright',
# #                   kind='line',
# #                   hue='sport', 
# #                   col='sport',
# #                   col_wrap=3)

# # g = sns.FacetGrid(Sport_duration_s_1, col="sport", row="sport")
# # g.map_dataframe(sns.histplot, x="start_time_hour", binwidth=2, binrange=(0, 60))

# # rel.fig.subplots_adjust(top=.95)

# # rel.fig.suptitle("Time vs Sport (24 hr)")

# # for ax in rel.fig.axes:
# #     ax.set_xlim(2,22)
# #     ax.set_xticks(range(2,22,2))

# # rel.set(xlabel="Hourly", ylabel = "Avg of Duration")
