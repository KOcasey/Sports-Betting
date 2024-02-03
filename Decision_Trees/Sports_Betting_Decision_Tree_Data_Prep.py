# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 11:09:18 2023

@author: casey
"""

## LOAD LIBRARIES

import pandas as pd

# --------------------------------------------------------------------------------------- #
## LOAD DATA

dt_df = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/data/gbg_cleaned.csv')

# --------------------------------------------------------------------------------------- #
## SELECT COLUMNS FOR DECISION TREE MODELING

cols = ['total_result', 'total_line', 'temp', 'wind', 'total_qb_elo', 'team_elo_diff',
        'qb_elo_diff', 'avg_home_total_yards', 'avg_away_total_yards', 
        'avg_home_total_yards_against', 'avg_away_total_yards_against', 'roof', 'surface',
        'div_game']

dt_df = dt_df[cols]

# --------------------------------------------------------------------------------------- #
## ONE HOT ENCODING OF qualitative CATEGORICAL VARIABLES (roof, surface)

labels = dt_df['total_result']
dt_df.drop(['total_result'], inplace=True, axis=1)
dt_df = pd.get_dummies(dt_df)
dt_df['total_result'] = labels

# --------------------------------------------------------------------------------------- #
## WRITE PREPPED DATA TO A .CSV FILE

dt_df.to_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Decision_Trees/prepped_data/gbg_dt.csv',
              index=False)