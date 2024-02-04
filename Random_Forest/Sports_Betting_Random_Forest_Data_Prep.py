# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 14:35:43 2023

@author: casey
"""

## LOAD LIBRARIES

import pandas as pd

# --------------------------------------------------------------------------------------- #
## LOAD DATA

rf_df = pd.read_csv('C:/Users/casey/OneDrive/Documents/Data_Science/Projects/Sports-Betting/data/gbg_cleaned.csv')

# --------------------------------------------------------------------------------------- #
## SELECT COLUMNS FOR RANDOM FOREST MODELING

cols = ['total_result', 'total_line', 'temp', 'wind', 'total_qb_elo', 'team_elo_diff',
        'qb_elo_diff', 'avg_home_total_yards', 'avg_away_total_yards',
        'avg_home_total_yards_against', 'avg_away_total_yards_against', 'roof', 'surface',
        'div_game']

rf_df = rf_df[cols]

# --------------------------------------------------------------------------------------- #
## ONE HOT ENCODING OF qualitative CATEGORICAL VARIABLES (roof, surface)
# since I will use min max scaling between 0 and 1 it's okay to one hot encode before normalization

labels = rf_df['total_result']
rf_df.drop(['total_result'], inplace=True, axis=1)
rf_df = pd.get_dummies(rf_df, dtype=int)
rf_df['total_result'] = labels


# --------------------------------------------------------------------------------------- #
## WRITE PREPPED DATA TO A .CSV FILE

rf_df.to_csv('C:/Users/casey/OneDrive/Documents/Data_Science/Projects/Sports-Betting/Random_Forest/prepped_data/gbg_rf.csv',
              index=False)