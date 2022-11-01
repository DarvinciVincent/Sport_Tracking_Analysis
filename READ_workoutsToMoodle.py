# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:06:17 2022

@author: h17163
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm



from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn import linear_model

df_res = pd.read_csv("data/df_res_filled.csv")

#df_res.to_csv("data/df_res_sorted.csv", index = False)
# # Define helper function for data loading and locating to one data frame
# def read_file_to_df(filename):
#     """ Created by h17163,
#         start and end latitude and longitude loading modified by Thanh Tran
#     """
#     data = pd.read_json(filename, typ='series')
#     value = []
#     key = []
#     for j in list(range(0, data.size)):
#         if list(data[j].keys())[0] != 'points':
#             key.append(list(data[j].keys())[0])
#             value.append(list(data[j].items())[0][1])
#             dictionary = dict(zip(key, value))
       

#     if list(data[j].keys())[0] == 'points':
#         try:
#             # start = list(list(list(data[data.size-1].items()))[0][1][0][0].items())[0][1][0]
#             # dictionary['start_lat'] = list(start[0].items())[0][1]
#             # dictionary['start_long'] = list(start[1].items())[0][1]
#             # dictionary['end_lat'] = list(start[0].items())[0][1]
#             # dictionary['end_long'] = list(start[1].items())[0][1]
#             points = list(data[j].items())[0][1]
#             start = points[0]
#             end = points[-1]
#             start_loc = start[0]['location']
#             end_loc = end[0]['location']
#             dictionary['start_lat'] = start_loc[0][0]['latitude']
#             dictionary['start_long'] = start_loc[0][1]['longitude']
#             dictionary['end_lat'] = end_loc[0][0]['latitude']
#             dictionary['end_long'] = end_loc[0][1]['longitude']
            
#         except:
#             print('No detailed data recorded')
            
        
#     df = pd.DataFrame(dictionary, index = [0])

#     return df

# # Load JSON data files from data folder @author: h17163
# folder = 'WorkoutData_2017to2020'
# file_list = os.listdir(folder)

# # Create Empty DataFrame @author: h17163
# df_res = pd.DataFrame()

# # Read files to a common dataframe @author: h17163
# for filename in file_list:
#     print('\n'+ filename)
#     df_process = read_file_to_df(folder + '/' + filename)
#     df_res = pd.concat([df_res, df_process], 0)

# df_res.reset_index(drop=True, inplace = True)

# # Save loaded data frame into "./data/df_res.csv"
# df_res.to_csv('data/df_res.csv', index = False)

# Read in data from the csv file and store it in the data matrix df_rex.
#df_res = pd.read_csv("./data/df_res.csv")

# Display first 5 rows
print("First five datapoints:")
display(df_res.head(5))

#%% START HERE
#df_res = pd.read_csv('data/df_res.csv')
#%% DATA EXPLORATION

#%% Observe the number of NaN values in df_res
# Checking Nan Values

NaN_check = {}
for i in df_res.columns:
    check = df_res[i].isnull().values.any()
    if check == True:
        NaN_check[i] = df_res[i].isnull().sum()
    else:
        NaN_check[i] = 0
# no operations on var with many NaN, fill 1 existing NaN in var calories
# Fill 1 existing NaN in var calories
# observe which sport corresponds the NaN
# Compute calory burn rate of that particular sport = calories / duration
# Compute mean calory burn rate and replace NaN

calories_burned_rate = df_res["calories_kcal"] / df_res["duration_s"]
NaN_index = df_res.loc[pd.isna(df_res["calories_kcal"]), :].index
print(f"The index of NAN value in column 'alories_kcal' is {NaN_index[0]}.")
df_res.loc[NaN_index[0],"calories_kcal"] = np.mean(calories_burned_rate) * df_res.loc[NaN_index[0],"duration_s"]


#%% DATA PRETREATMENT

# Create dummy variables from categorial variable (source, sport)
sport_dummy = pd.get_dummies(df_res["sport"],prefix="sport")
source_dummy = pd.get_dummies(df_res["source"],prefix="source")

# Convert datetime variables to int
df_res["created_date_num"] = pd.to_datetime(df_res["created_date"])
df_res["created_date_num"] = df_res["created_date_num"].apply(lambda x: x.strftime("%Y%m%d%H%M%S"))
df_res["created_date_num"] = df_res["created_date_num"].apply(lambda x: float(x))

df_res["start_timee_num"] = pd.to_datetime(df_res["start_time"])
df_res["start_timee_num"] = df_res["start_timee_num"].apply(lambda x: x.strftime("%Y%m%d%H%M%S"))
df_res["start_timee_num"] = df_res["start_timee_num"].apply(lambda x: float(x))

df_res["end_time_num"] = pd.to_datetime(df_res["end_time"])
df_res["end_time_num"] = df_res["end_time_num"].apply(lambda x: x.strftime("%Y%m%d%H%M%S"))
df_res["end_time_num"] = df_res["end_time_num"].apply(lambda x: float(x))



#%% SOLUTION 1:
# Create dataset for solution 1 => DF1 (9 columns)
# Contains variables without NaN
NAN_check_1 = NaN_check
NAN_check_1["calories_kcal"] = 0
column_remove = []
for i in NAN_check_1:
    if NAN_check_1[i] != 0:
        column_remove.append(i)

# Remove columns including NaN value more than 1
DF1 = df_res[df_res.columns[~df_res.columns.isin(column_remove)]]

# DF1_class
DF1_sport = DF1["sport"]
nObs = sport_dummy.shape[0]
# DF1_class = sport_dummy.iloc[7:]
DF1_class = DF1_sport.iloc[7:]
DF1_class = DF1_class.apply(lambda x: str(x))
DF1_class = DF1_class.to_numpy()
DF1_class = DF1_class.astype(np.str)


del DF1["created_date"]
del DF1["start_time"]
del DF1["end_time"]
del DF1["sport"]
del DF1["source"]


DF1 = pd.concat([DF1, source_dummy, sport_dummy], axis = 1)

# Min-Max normalization
DF1["duration_s"] = (DF1["duration_s"]-DF1["duration_s"].min())/(DF1["duration_s"].max()-DF1["duration_s"].min())
DF1["distance_km"] = (DF1["distance_km"]-DF1["distance_km"].min())/(DF1["distance_km"].max()-DF1["distance_km"].min())
DF1["calories_kcal"] = (DF1["calories_kcal"]-DF1["calories_kcal"].min())/(DF1["calories_kcal"].max()-DF1["calories_kcal"].min())
DF1["speed_avg_kmh"] = (DF1["speed_avg_kmh"]-DF1["speed_avg_kmh"].min())/(DF1["speed_avg_kmh"].max()-DF1["speed_avg_kmh"].min())
DF1["created_date_num"] = (DF1["created_date_num"]-DF1["created_date_num"].min())/(DF1["created_date_num"].max()-DF1["created_date_num"].min())
DF1["start_timee_num"] = (DF1["start_timee_num"]-DF1["start_timee_num"].min())/(DF1["start_timee_num"].max()-DF1["start_timee_num"].min())
DF1["end_time_num"] = (DF1["end_time_num"]-DF1["end_time_num"].min())/(DF1["end_time_num"].max()-DF1["end_time_num"].min())

DF1_data = pd.DataFrame()
# Create shifted dataset including past events as additional columns
for j in DF1.columns:
    for i in range(1,8): 
        name = j+"_"+str(i)
        DF1_data[name] = DF1[j].shift(i)
        
DF1_data = DF1_data.iloc[7:,:]
DF1_data = DF1_data.to_numpy()

#%% Construct model for next SPORT prediction (KNN, CLUSTERING K-MEANS, GAUSSIAN MODELS) using DF1_1
X_train, X_test, y_train, y_test = train_test_split(DF1_data, DF1_class, random_state=42, test_size=0.3)
Acc_lst = {}
for i in range(3,30,2):
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(X_train, y_train)
    y_pred_5 = knn_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_5)*100
    Acc_lst[i] = acc

best_acc = max(Acc_lst)

s1_sport_model = KNeighborsClassifier(n_neighbors=11)
s1_sport_model.fit(X_train, y_train)
y_pred_11 = s1_sport_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_11)*100

print(f"Accuracy with k=11 is {accuracy}")

#%% SOLUTION 1_1:

# Create dataset for solution 1 => DF1 (9 columns)
# Contains variables without NaN
# NAN_check_1 = NaN_check
# NAN_check_1["calories_kcal"] = 0
# column_remove = []
# for i in NAN_check_1:
#     if NAN_check_1[i] != 0:
#         column_remove.append(i)

# # Remove columns including NaN value more than 1
# DF1_1 = df_res[df_res.columns[~df_res.columns.isin(column_remove)]]


# DF1_1 = pd.concat([DF1_1, source_dummy, sport_dummy], axis = 1)

# del DF1_1["sport"]
# del DF1_1["source"]



# # Min-Max normalization
# DF1_1["duration_s"] = (DF1_1["duration_s"]-DF1_1["duration_s"].min())/(DF1_1["duration_s"].max()-DF1_1["duration_s"].min())
# DF1_1["distance_km"] = (DF1_1["distance_km"]-DF1_1["distance_km"].min())/(DF1_1["distance_km"].max()-DF1_1["distance_km"].min())
# DF1_1["calories_kcal"] = (DF1_1["calories_kcal"]-DF1_1["calories_kcal"].min())/(DF1_1["calories_kcal"].max()-DF1_1["calories_kcal"].min())
# DF1_1["speed_avg_kmh"] = (DF1_1["speed_avg_kmh"]-DF1_1["speed_avg_kmh"].min())/(DF1_1["speed_avg_kmh"].max()-DF1_1["speed_avg_kmh"].min())
# DF1_1["start_timee_num"] = (DF1_1["start_timee_num"]-DF1_1["start_timee_num"].min())/(DF1_1["start_timee_num"].max()-DF1_1["start_timee_num"].min())
# DF1_1["end_time_num"] = (DF1_1["end_time_num"]-DF1_1["end_time_num"].min())/(DF1_1["end_time_num"].max()-DF1_1["end_time_num"].min())
# DF1_1["created_date_num"] = (DF1_1["created_date_num"]-DF1_1["created_date_num"].min())/(DF1_1["created_date_num"].max()-DF1_1["created_date_num"].min())

# DF1_1_data = pd.DataFrame()
# # Create shifted dataset including past events as additional columns
# for j in DF1_1.columns:
#     for i in range(1,8): 
#         name = j+"_"+str(i)
#         DF1_1_data[name] = DF1_1[j].shift(i)
        
# # DF1_1_class
# DF1_1_time = DF1_1["created_date_num"]
# nObs = sport_dummy.shape[0]
# DF1_1_class = DF1_1_time.iloc[7:]
# DF1_1_class = DF1_1_class.apply(lambda x: float(x))
# DF1_1_class = DF1_1_class.to_numpy()
# DF1_1_class = DF1_1_class.astype(np.float)

# del DF1_1["duration_s"]
# del DF1_1["created_date"]
# del DF1_1["start_time"]
# del DF1_1["end_time"]   
# del DF1_1["end_time_num"]
# del DF1_1["created_date_num"]
# del DF1_1["start_timee_num"]
# del DF1_1["distance_km"]
# del DF1_1["calories_kcal"]
# del DF1_1["speed_avg_kmh"]


        
# DF1_1_data = DF1_1_data.iloc[7:,:]
# DF1_1_data_nump = DF1_1_data.to_numpy()

#%% Construct model for next TIME prediction (Ridge) using DF1_1
# X_train, X_test, y_train, y_test = train_test_split(DF1_1_data_nump, DF1_1_class, random_state=42, test_size=0.3)

# model = Ridge(alpha = 1)
# model.fit(X_train, y_train)

# coefficient  = model.score(X_train, y_train)
# print(f"coefficient of determination: {coefficient}")

# print(f"intercept: {model.intercept_}")

# print(f"slope: {model.coef_}")

# y_pred = model.predict(X_test)
# print(f"predicted response:\n{y_pred}")

# accuracy = r2_score(y_test, y_pred)
# print(f"Accuracy is {accuracy}")

# mean_absolut_error = mean_absolute_error(y_test, y_pred)
# print(f"mean_absolute_error is {mean_absolut_error}")

# mean_square_error = mean_squared_error(y_test, y_pred)
# print(f"mean_squared_error is {mean_square_error}")

# pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# display(pred_df)

#%% Construct model for next TIME prediction (LINEAR REGRESSION) using DF1_1

# X_train, X_test, y_train, y_test = train_test_split(DF1_1_data_nump, DF1_1_class, random_state=42, test_size=0.3)

# model = LinearRegression().fit(X_train, y_train)

# coefficient  = model.score(X_train, y_train)
# print(f"coefficient of determination: {coefficient}")

# print(f"intercept: {model.intercept_}")

# print(f"slope: {model.coef_}")

# y_pred = model.predict(X_test)
# print(f"predicted response:\n{y_pred}")

# accuracy = r2_score(y_test, y_pred)
# print(f"Accuracy is {accuracy}")

# mean_absolut_error = mean_absolute_error(y_test, y_pred)
# print(f"mean_absolute_error is {mean_absolut_error}")

# mean_square_error = mean_squared_error(y_test, y_pred)
# print(f"mean_squared_error is {mean_square_error}")

# pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
display(pred_df)

#%% SOLUTION 1_2:

# Create dataset for solution 1 => DF1 (9 columns)
# Contains variables without NaN
# NAN_check_1 = NaN_check
# NAN_check_1["calories_kcal"] = 0
# column_remove = []
# for i in NAN_check_1:
#     if NAN_check_1[i] != 0:
#         column_remove.append(i)

# # Remove columns including NaN value more than 1
# DF1_2 = df_res[df_res.columns[~df_res.columns.isin(column_remove)]]


# DF1_2 = pd.concat([DF1_2, source_dummy, sport_dummy], axis = 1)

# # Min-Max normalization
# DF1_2["duration_s"] = (DF1_2["duration_s"]-DF1_2["duration_s"].min())/(DF1_2["duration_s"].max()-DF1_2["duration_s"].min())
# DF1_2["distance_km"] = (DF1_2["distance_km"]-DF1_2["distance_km"].min())/(DF1_2["distance_km"].max()-DF1_2["distance_km"].min())
# DF1_2["calories_kcal"] = (DF1_2["calories_kcal"]-DF1_2["calories_kcal"].min())/(DF1_2["calories_kcal"].max()-DF1_2["calories_kcal"].min())
# DF1_2["speed_avg_kmh"] = (DF1_2["speed_avg_kmh"]-DF1_2["speed_avg_kmh"].min())/(DF1_2["speed_avg_kmh"].max()-DF1_2["speed_avg_kmh"].min())
# DF1_2["start_timee_num"] = (DF1_2["start_timee_num"]-DF1_2["start_timee_num"].min())/(DF1_2["start_timee_num"].max()-DF1_2["start_timee_num"].min())
# DF1_2["end_time_num"] = (DF1_2["end_time_num"]-DF1_2["end_time_num"].min())/(DF1_2["end_time_num"].max()-DF1_2["end_time_num"].min())
# DF1_2["created_date_num"] = (DF1_2["created_date_num"]-DF1_2["created_date_num"].min())/(DF1_2["created_date_num"].max()-DF1_2["created_date_num"].min())



# DF1_2_data = pd.DataFrame()
# # Create shifted dataset including past events as additional columns
# for j in DF1_2.columns:
#     for i in range(1,8): 
#         name = j+"_"+str(i)
#         DF1_2_data[name] = DF1_2[j].shift(i)

# # DF1_2_class
# DF1_2_time = DF1_2["duration_s"]
# nObs = sport_dummy.shape[0]
# DF1_2_class = DF1_2_time.iloc[7:]
# DF1_2_class = DF1_2_class.apply(lambda x: float(x))
# DF1_2_class = DF1_2_class.to_numpy()
# DF1_2_class = DF1_2_class.astype(np.float)


# del DF1_2["created_date"]
# del DF1_2["start_time"]
# del DF1_2["end_time"]
# del DF1_2["sport"]
# del DF1_2["source"]
# del DF1_2["created_date_num"]
# del DF1_2["start_timee_num"]
# del DF1_2["end_time_num"]
# del DF1_2["duration_s"]
        
# DF1_2_data = DF1_2_data.iloc[7:,:]
# DF1_2_data_nump = DF1_2_data.to_numpy()

#%% Construct model for next DURATION prediction (LINEAR REGRESSION) using DF1_2
















