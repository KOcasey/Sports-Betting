# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 08:55:19 2023

@author: casey
"""

## LOAD LIBRARIES

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------------------------------------- #
## LOAD DATA

lr_df = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/data/gbg_cleaned.csv')

# --------------------------------------------------------------------------------------- #
## SELECT COLUMNS FOR Logistic Regression MODELING

cols = ['total_result', 'total_line', 'temp', 'wind', 'total_qb_elo', 'team_elo_diff',
        'qb_elo_diff', 'avg_home_total_yards', 'avg_away_total_yards', 
        'avg_home_total_yards_against', 'avg_away_total_yards_against', 'roof', 'surface',
        'div_game']

lr_df = lr_df[cols]

# --------------------------------------------------------------------------------------- #
## ONE HOT ENCODING OF qualitative CATEGORICAL VARIABLES (roof, surface)
# since I will use min max scaling between 0 and 1 it's okay to one hot encode before normalization

labels = lr_df['total_result']
lr_df.drop(['total_result'], inplace=True, axis=1)
lr_df = pd.get_dummies(lr_df)
lr_df['total_result'] = labels

# --------------------------------------------------------------------------------------- #
## NORMALIZE DATA USING MINMAXSCALER 
# Will use MinMaxScaler() to scale all quantitative variables between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
lr_df_scaled = scaler.fit_transform(lr_df.iloc[:,:-1])
lr_df_scaled = pd.DataFrame(lr_df_scaled, columns=lr_df.iloc[:,:-1].columns)
lr_df_scaled['total_result'] = lr_df['total_result']

# --------------------------------------------------------------------------------------- #
## WRITE PREPPED DATA TO A .CSV FILE

lr_df_scaled.to_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Logistic_Regression/prepped_data/gbg_lr_py.csv',
              index=False)