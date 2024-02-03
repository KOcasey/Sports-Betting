# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:39:54 2023

@author: casey
"""

## LOAD LIBRARIES

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --------------------------------------------------------------------------------------- #
## LOAD DATA

svm_df = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/data/gbg_cleaned.csv')


# --------------------------------------------------------------------------------------- #
## SELECT COLUMNS FOR SVM MODELING
# only select quantitative variables and label variable

cols = ['total_result', 'total_line', 'temp', 'wind', 'total_qb_elo', 'team_elo_diff',
        'qb_elo_diff', 'avg_home_total_yards', 'avg_away_total_yards', 
        'avg_home_total_yards_against', 'avg_away_total_yards_against']

svm_df = svm_df[cols]

# create new variables to get the difference in yards between home team offense and away team defense
# positive number favors home team offense, negative number favors away team defense
svm_df['home_away_yards_diff'] = svm_df['avg_home_total_yards'] - svm_df['avg_away_total_yards_against']

# create new variable to get the difference in yards between away team offense and home team defense
# positive number favors away team offense, negative number favors home team defense
svm_df['away_home_yards_diff'] = svm_df['avg_away_total_yards'] - svm_df['avg_home_total_yards_against']

# --------------------------------------------------------------------------------------- #
## CHECK FOR ANY MISSING VALUES

# should be no missing values
svm_df.isna().sum()

# --------------------------------------------------------------------------------------- #
## NORMALIZE DATA USING MINMAXSCALER 
# Will use MinMaxScaler() to scale all quantitative variables between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
svm_df_scaled = scaler.fit_transform(svm_df.iloc[:,1:])
svm_df_scaled = pd.DataFrame(svm_df_scaled, columns=svm_df.iloc[:,1:].columns)
svm_df_scaled['total_result'] = svm_df['total_result']


# --------------------------------------------------------------------------------------- #
## WRITE PREPPED SVM DATA TO A .CSV FILE

svm_df_scaled.to_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/SVM/prepped_data/gbg_svm_py.csv',
              index=False)
