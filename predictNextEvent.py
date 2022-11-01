# -*- coding: utf-8 -*-
"""
Author: Thanh Tran (000285359)

PRACTICAL ASSIGNMENT
"""

# Import required libraries
import os
import string
import re
import joblib

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import HuberRegressor, LogisticRegression

def read_file_to_df(filename):
    """ Created by h17163,
        start and end latitude and longitude loading modified by Thanh Tran
    """
    data = pd.read_json(filename, typ='series')
    value = []
    key = []
    for j in list(range(0, data.size)):
        if list(data[j].keys())[0] != 'points':
            key.append(list(data[j].keys())[0])
            value.append(list(data[j].items())[0][1])
            dictionary = dict(zip(key, value))
       

    if list(data[j].keys())[0] == 'points':
        try:
            points = list(data[j].items())[0][1]
            start = points[0]
            end = points[-1]
            start_loc = start[0]['location']
            end_loc = end[0]['location']
            dictionary['start_lat'] = start_loc[0][0]['latitude']
            dictionary['start_long'] = start_loc[0][1]['longitude']
            dictionary['end_lat'] = end_loc[0][0]['latitude']
            dictionary['end_long'] = end_loc[0][1]['longitude']
            
        except:
            print('No detailed data recorded')
            
        
    df = pd.DataFrame(dictionary, index = [0])

    return df

def extractDatetime(df, ColName):
    df[f"{ColName}_year"] = df[ColName].apply(lambda x: int(x.strftime("%Y")))

    df[f"{ColName}_month"] = df[ColName].apply(lambda x: int(x.strftime("%m")))

    df[f"{ColName}_day"] = df[ColName].apply(lambda x: int(x.strftime("%d")))

    df[f"{ColName}_weekday_cat"] = df[ColName].apply(lambda x: x.strftime("%A"))
    df[f"{ColName}_weekday_cat"] = df[f"{ColName}_weekday_cat"].astype('string')

    df[f"{ColName}_hour"] = df[ColName].apply(lambda x: int(x.strftime("%H")))

    df[f"{ColName}_minute"] = df[ColName].apply(lambda x: int(x.strftime("%M")))

    df[f"{ColName}_second"] = df[ColName].apply(lambda x: int(x.strftime("%S")))

    num_cols = [f"{ColName}_year", f"{ColName}_month", f"{ColName}_day", f"{ColName}_hour", f"{ColName}_minute", f"{ColName}_second"]
    cat_cols = [f"{ColName}_weekday_cat"]
    
    return df, num_cols, cat_cols

def getDaytimeFromHour(df, ColName):
    # Define categoring thresholds
    morning_default_left = 5
    morning_default_right = 10


    afternoon_default_left = 11
    afternoon_default_right = 16

    evening_default_left = 17
    evening_default_right = 22

    night_default_left = 23
    night_default_right = 4

    # Categorize start_time
    df[f"{ColName}_daytime_cat"] = df[f"{ColName}_hour"]

    for i in range(0,len(df[f"{ColName}_hour"])):
        if morning_default_left <= df.loc[i, f"{ColName}_hour"] <= morning_default_right:
            df.loc[i, f"{ColName}_daytime_cat"] = "MORNING"
        elif afternoon_default_left <= df.loc[i, f"{ColName}_hour"] <= afternoon_default_right:
            df.loc[i, f"{ColName}_daytime_cat"] = "AFTERNOON"
        elif evening_default_left <= df.loc[i, f"{ColName}_hour"] <= evening_default_right:
            df.loc[i, f"{ColName}_daytime_cat"] = "EVENING"
        elif night_default_left <= df.loc[i, f"{ColName}_hour"] <= 24 or 0 <= df.loc[i, f"{ColName}_hour"] <= night_default_right:
            df.loc[i, f"{ColName}_daytime_cat"] = "NIGHT"
        else:
            df.loc[i, f"{ColName}_daytime_cat"] = np.NaN
            
    df[f"{ColName}_daytime_cat"] = df[f"{ColName}_daytime_cat"].astype('string')

    daytimeColName = f"{ColName}_daytime_cat"
    
    return df, daytimeColName

def getNorthernSeasonFromMonth(df, ColName):

    # Define categoring thresholds
    winter_default_left = 12
    winter_default_right = 2


    spring_default_left = 3
    spring_default_right = 5

    summer_default_left = 6
    summer_default_right = 8

    autumn_default_left = 9
    autumn_default_right = 11

    # Categorize start_time
    df[f"{ColName}_season_cat"] = df[f"{ColName}_month"]

    for i in range(0,len(df[f"{ColName}_month"])):
        if spring_default_left <= df.loc[i, f"{ColName}_month"] <= spring_default_right:
            df.loc[i, f"{ColName}_season_cat"] = "SPRING"
        elif summer_default_left <= df.loc[i, f"{ColName}_month"] <= summer_default_right:
            df.loc[i, f"{ColName}_season_cat"] = "SUMMER"
        elif autumn_default_left <= df.loc[i, f"{ColName}_month"] <= autumn_default_right:
            df.loc[i, f"{ColName}_season_cat"] = "AUTUMN"
        elif winter_default_left <= df.loc[i, f"{ColName}_month"] <= 12 or 0 <= df.loc[i, f"{ColName}_month"] <= winter_default_right:
            df.loc[i, f"{ColName}_season_cat"] = "WINTER"
        else:
            df.loc[i, f"{ColName}_season_cat"] = np.NaN

    df[f"{ColName}_season_cat"] = df[f"{ColName}_season_cat"].astype('string')

    seasonColName = f"{ColName}_season_cat"
    
    return df, seasonColName

def min_max_normalize_with_min_max_values(df, min_max_values_df, reallocate = True):
    Cols = df.columns.to_list()

    numeric_types = ["float", "float64", "int", "int64", "uint8"]
    Cols_Idx = df.shape[1]
    skipped_cols = []
    normalized_cols = []

    for c in range(Cols_Idx):
        if df.iloc[:,c].dtype in numeric_types:
            # Get min and max values
            col_name = Cols[c]
            rowIdx = min_max_values_df.index[min_max_values_df['Columns Name'] == col_name]
            min_val = float(min_max_values_df.loc[rowIdx, 'min_values'])
            max_val = float(min_max_values_df.loc[rowIdx, 'max_values'])
            df.iloc[:,c] = (df.iloc[:,c]-min_val)/(max_val-min_val)
            normalized_cols.append(c)

        else:
            skipped_cols.append(c)

    if reallocate:
        df = df.iloc[:, skipped_cols + normalized_cols]

    return df

def past_time_shift(df_, classCol, nEvents, to_numpy_ = True, Xfilename=None, Yfilename=None):
    df_Y = df_[classCol]
    df_X = df_.drop([classCol], axis = 1)
    Cols_X = list(df_X.columns)
    
    for c in Cols_X:
        for e in range(1, nEvents + 1): 
            name = c+"_"+str(e)
            df_X[name] = df_X[c].shift(e)
    
    # Keep only past event columns
    df_X = df_X.drop(Cols_X, axis = 1)
    # Remove nEvents first rows having NaN
    df_X = df_X.iloc[nEvents:, :]
    df_Y = df_Y.iloc[nEvents:]

    df_X.reset_index(drop = True, inplace = True)
    df_Y.reset_index(drop = True, inplace = True)

    if to_numpy_:
        # Transform to numpy array
        df_X = df_X.to_numpy()
        df_Y = df_Y.to_numpy()

        if Xfilename:
            np.savetxt(Xfilename, df_X, delimiter = ",")
        if Yfilename:
            if isinstance(df_Y[0], str):
                np.savetxt(Yfilename, df_Y, delimiter = ",", fmt='%s')
            else:
                np.savetxt(Yfilename, df_Y, delimiter = ",")
    else:
        if Xfilename:
            df_X.to_csv(Xfilename, index=False)
        if Xfilename:
            df_Y.to_csv(Yfilename, index=False)

    return df_X, df_Y

def predictNextEvent(jsonFileList):
    DF_SPORT_Y_pred, DF_SPORT_Y, DF_DAYTIME_Y_pred, DF_DAYTIME_Y, DF_HOUR_Y_pred, DF_HOUR_Y, DF_DUR_Y_pred, DF_DUR_Y = None, None, None, None, None, None, None, None
    # Load JSON data files from data folder
    df_res = pd.DataFrame()
    
    # Read files to a common dataframe
    print("Data loading ...........................................................................")
    for filename in jsonFileList:
        print('\n'+ filename)
        df_process = read_file_to_df(folder + '/' + filename)
        df_res = pd.concat([df_res, df_process], 0)
        
    df_res.reset_index(drop=True, inplace = True)
    
    print("Data processing ...........................................................................")
    # Sort by start time
    # Convert time variable to datetime format
    df_res["start_time"] = pd.to_datetime(df_res["start_time"])

    # Sort dataset by variable "start_time" and reset index
    df_res.sort_values(by="start_time", ascending = True, inplace = True)
    df_res.reset_index(drop = True, inplace = True)
    
    # Convert object-typed variables to string
    df_res["sport"] = df_res["sport"].astype('string')
    df_res["source"] = df_res["source"].astype('string')
    
    # Removed cols
    df_res_cols = df_res.columns.to_list()
    
    excluded_cols = ["source", "created_date", "end_time", "start_lat", "start_long", "end_lat", "end_long", "speed_max_kmh", "altitude_min_m", "altitude_max_m", "ascend_m", "descend_m"]
    
    if "hydration_l" in df_res_cols:
        excluded_cols.append("hydration_l")
    
    df_res = df_res.drop(excluded_cols, axis=1)
    
    # Check NaN
    # Count NaN values of each feature
    df_res_nNaN = dict()  # Initialize list of number of NaN values in corresponding columns
    df_res_cols = list(df_res.columns)  # columns of df_res
    [df_res_nObs, df_res_nCol] = df_res.shape
    
    for col in df_res_cols:
        df_res_nNaN[col] = df_res[col].isnull().sum()
    
    if sum(df_res_nNaN.values()) > 0:
        print("Model cannot predict because of NaN existence")
    else:
        df_res, start_time_num_cols, start_time_cat_cols = extractDatetime(df_res, "start_time")
        # Extract categorical daytime from "start_time"
        df_res, start_time_daytime_catColName = getDaytimeFromHour(df_res, "start_time")
        # Extract categorical seasons from "start_time"
        df_res, start_time_season_catColName = getNorthernSeasonFromMonth(df_res, "start_time")
        start_time_cat_cols = start_time_cat_cols + [start_time_daytime_catColName, start_time_season_catColName]
        
        # Create dummy for "sport" variable
        sport_dummy_cols = ['sport_BADMINTON', 'sport_BEACH_VOLLEY', 'sport_CROSSFIT', 'sport_CROSS_TRAINING', 'sport_CYCLING_SPORT', 'sport_CYCLING_TRANSPORTATION', 'sport_FITNESS_WALKING', 'sport_ICE_SKATING', 'sport_ROLLER_SKATING', 'sport_RUNNING', 'sport_RUNNING_CANICROSS', 'sport_SKIING_CROSS_COUNTRY', 'sport_STAIR_CLIMBING', 'sport_STRETCHING', 'sport_SWIMMING', 'sport_WALKING', 'sport_WEIGHT_TRAINING']
        
        for c in sport_dummy_cols:
            df_res[c] = 0
            
        # Create dummy for "start_time_weekday_cat" variable
        start_time_weekday_dummy_cols = ['start_time_weekday_Friday', 'start_time_weekday_Monday', 'start_time_weekday_Saturday', 'start_time_weekday_Sunday', 'start_time_weekday_Thursday', 'start_time_weekday_Tuesday', 'start_time_weekday_Wednesday']
        
        for c in start_time_weekday_dummy_cols:
            df_res[c] = 0
            
        # Create dummy for "start_time_daytime_cat" variable
        start_time_daytime_dummy_cols = ['start_time_daytime_AFTERNOON', 'start_time_daytime_EVENING', 'start_time_daytime_MORNING', 'start_time_daytime_NIGHT']
        
        for c in start_time_daytime_dummy_cols:
            df_res[c] = 0
            
        # Create dummy for "start_time_season_cat" variable
        start_time_season_dummy_cols = ['start_time_season_AUTUMN', 'start_time_season_SPRING', 'start_time_season_SUMMER', 'start_time_season_WINTER']
        
        for c in start_time_season_dummy_cols:
            df_res[c] = 0
            
        ### Create dummies
        for rIdx in range(len(df_res)):
            # Fix dummy sport
            rSport = df_res.loc[rIdx, "sport"]
            df_res.loc[rIdx, f"sport_{rSport}"] = 1
            # Fix dummy weekday
            rWeekday = df_res.loc[rIdx, "start_time_weekday_cat"]
            df_res.loc[rIdx, f"start_time_weekday_{rWeekday}"] = 1
            # Fix dummy daytime
            rDaytime = df_res.loc[rIdx, "start_time_daytime_cat"]
            df_res.loc[rIdx, f"start_time_daytime_{rDaytime}"] = 1
            # Fix dummy season
            rSeason = df_res.loc[rIdx, "start_time_season_cat"]
            df_res.loc[rIdx, f"start_time_season_{rSeason}"] = 1
        
        # Normalize
        min_max_values_df = pd.read_csv('data/min_max_values.csv')
        df_norm = df_res.copy()
        df_norm = min_max_normalize_with_min_max_values(df_norm, min_max_values_df)
        
        #Remove additional cols
        additional_excluded_cols = ["start_time"] + start_time_num_cols + [start_time_season_catColName] + ["start_time_weekday_cat"]
        df_norm = df_norm.drop(additional_excluded_cols, axis=1)
        
        # Generate datasets
        SPORT_CLASS_COL = "sport"
        DAYTIME_CLASS_COL = "start_time_daytime_cat"
        HOUR_CLASS_COL = "start_time_hour_original"
        DUR_CLASS_COL = "duration_s_original"

        DF_SPORT = df_norm.drop([DAYTIME_CLASS_COL], axis=1)
        
        DF_DAYTIME = df_norm.drop([SPORT_CLASS_COL], axis=1)
        
        DF_HOUR = df_norm.drop([start_time_daytime_catColName, "sport"], axis=1)
        DF_HOUR_Cols = DF_HOUR.columns.to_list()
        DF_HOUR[HOUR_CLASS_COL] = df_res["start_time_hour"]
        DF_HOUR = DF_HOUR.loc[:, [HOUR_CLASS_COL] + DF_HOUR_Cols]
        
        DF_DUR = df_norm.drop([start_time_daytime_catColName, "sport"], axis=1)
        DF_DUR_Cols = DF_DUR.columns.to_list()
        DF_DUR[DUR_CLASS_COL] = df_res["duration_s"]
        DF_DUR = DF_DUR.loc[:, [DUR_CLASS_COL] + DF_DUR_Cols]
        
        ### Past event shifting
        
        nEvents = 15
        ### Dataset DF2_SPORT_X
        DF_SPORT_X, DF_SPORT_Y = past_time_shift(DF_SPORT, SPORT_CLASS_COL, nEvents, True)
        
        ### Dataset DF2_DAYTIME_X
        DF_DAYTIME_X, DF_DAYTIME_Y = past_time_shift(DF_DAYTIME, DAYTIME_CLASS_COL, nEvents, True)
        
        ### Dataset DF2_HOUR_X
        DF_HOUR_X, DF_HOUR_Y = past_time_shift(DF_HOUR, HOUR_CLASS_COL, nEvents, True)
        
        ### Dataset DF2_DUR_X
        DF_DUR_X, DF_DUR_Y = past_time_shift(DF_DUR, DUR_CLASS_COL, nEvents, True)
        
        ### Load trained models
        print("MODEL loading ...........................................................................")
        DF2_SPORT_MODEL_KNN_filename = "model/DF2_SPORT_MODEL_KNN.sav"
        DF2_DAYTIME_MODEL_LOG_filename = "model/DF2_DAYTIME_MODEL_LOG.sav"
        DF2_HOUR_MODEL_filename = "model/DF2_HOUR_MODEL.sav"
        DF2_DUR_MODEL_filename = "model/DF2_DUR_MODEL.sav"
        
        FINAL_SPORT_PREDICTION_MODEL = joblib.load(DF2_SPORT_MODEL_KNN_filename)
        FINAL_DAYTIME_PREDICTION_MODEL = joblib.load(DF2_DAYTIME_MODEL_LOG_filename)
        FINAL_HOUR_PREDICTION_MODEL = joblib.load(DF2_HOUR_MODEL_filename)
        FINAL_DURATION_PREDICTION_MODEL = joblib.load(DF2_DUR_MODEL_filename)
        
        ### Predict next event
        print("PREDICTING ...........................................................................")

        DF_SPORT_Y_pred = FINAL_SPORT_PREDICTION_MODEL.predict(DF_SPORT_X)
        print(f"SPORT: \n       Predicted: {DF_SPORT_Y_pred}\n       Actual: {DF_SPORT_Y}")
        
        DF_DAYTIME_Y_pred = FINAL_DAYTIME_PREDICTION_MODEL.predict(DF_DAYTIME_X)
        print(f"DAYTIME: \n       Predicted: {DF_DAYTIME_Y_pred}\n       Actual: {DF_DAYTIME_Y}")
        
        DF_HOUR_Y_pred = FINAL_HOUR_PREDICTION_MODEL.predict(DF_HOUR_X)
        print(f"HOUR: \n       Predicted: {DF_HOUR_Y_pred} (Mean error: ~3h)\n       Actual: {DF_HOUR_Y}")
        
        DF_DUR_Y_pred = FINAL_DURATION_PREDICTION_MODEL.predict(DF_DUR_X)
        print(f"DURATION: \n       Predicted: {DF_DUR_Y_pred} (Mean error: ~700s\n       Actual: {DF_DUR_Y}")
        
        print("FINISHED!")
        
    return DF_SPORT_Y_pred, DF_SPORT_Y, DF_DAYTIME_Y_pred, DF_DAYTIME_Y, DF_HOUR_Y_pred, DF_HOUR_Y, DF_DUR_Y_pred, DF_DUR_Y

# REQUIREMENT: MINIMUM 16 FILES MUST BE AVAILABLE
folder = 'WorkoutData_2017to2020'
file_list = os.listdir(folder)

# TEST PREDICTING FUNCTION
jsonFileList = file_list[:16]
DF_SPORT_Y_pred, DF_SPORT_Y, DF_DAYTIME_Y_pred, DF_DAYTIME_Y, DF_HOUR_Y_pred, DF_HOUR_Y, DF_DUR_Y_pred, DF_DUR_Y = predictNextEvent(jsonFileList)