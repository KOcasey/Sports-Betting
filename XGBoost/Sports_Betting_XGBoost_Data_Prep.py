# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 14:37:02 2023

@author: casey
"""

## LOAD LIBRARIES

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------------------- #
## LOAD DATA

xg_df = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/data/gbg_cleaned.csv')

# --------------------------------------------------------------------------------------- #
## SELECT COLUMNS FOR XGBoost MODELING

cols = ['total_result', 'total_line', 'temp', 'wind', 'total_qb_elo', 'team_elo_diff',
        'qb_elo_diff', 'avg_home_total_yards', 'avg_away_total_yards', 
        'avg_home_total_yards_against', 'avg_away_total_yards_against', 'roof', 'surface',
        'div_game']

xg_df = xg_df[cols]

# --------------------------------------------------------------------------------------- #
## CONVERT CATEGORICAL LABELS TO NUMERIC LABELS

le = LabelEncoder()
xg_df['total_result'] = le.fit_transform(xg_df['total_result'])

# --------------------------------------------------------------------------------------- #
## ONE HOT ENCODING OF qualitative CATEGORICAL VARIABLES (roof, surface)

labels = xg_df['total_result']
xg_df.drop(['total_result'], inplace=True, axis=1)
xg_df = pd.get_dummies(xg_df)
xg_df['total_result'] = labels

# --------------------------------------------------------------------------------------- #
## WRITE PREPPED DATA TO A .CSV FILE

xg_df.to_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/XGBoost/prepped_data/gbg_xg.csv',
              index=False)