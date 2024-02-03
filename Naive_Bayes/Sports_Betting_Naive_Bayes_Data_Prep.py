# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 14:15:23 2023

@author: casey
"""

## LOAD LIBRARIES
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Import visualization
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------------- #
## LOAD DATA

nb_df = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/data/gbg_cleaned.csv')

# --------------------------------------------------------------------------------------- #
## SELECT COLUMNS FOR NAIVE BAYES MODELING

cols = ['total_result', 'total_line', 'temp', 'wind', 'total_qb_elo', 'team_elo_diff',
        'qb_elo_diff', 'avg_home_total_yards', 'avg_away_total_yards', 
        'avg_home_total_yards_against', 'avg_away_total_yards_against', 'roof', 'surface',
        'div_game']

nb_df = nb_df[cols]

# ------------------------------------------------------------------------------------------------ #
## CHECK FOR VARIABLE CORRELATION
cor_df = nb_df.copy()

# convert categorical variables to number codes so they can be used in the heatmap
cor_df['roof'] = cor_df['roof'].astype('category')
cor_df['roof'] = cor_df['roof'].cat.codes
cor_df['surface'] = cor_df['surface'].astype('category')
cor_df['surface'] = cor_df['surface'].cat.codes


# heatmap plot
fig, ax = plt.subplots(figsize=(12,12)) 
dataplot = sns.heatmap(nb_df.corr(), cmap="YlGnBu", annot=True, ax=ax)
  
# displaying heatmap
plt.show()

# remove highly correlated columns from nb_df
nb_df.drop(columns=['avg_away_total_yards', 'avg_home_total_yards_against', 'avg_away_total_yards_against'], axis=1, inplace=True)

# --------------------------------------------------------------------------------------- #
## ONE HOT ENCODING OF CATEGORICAL VARIABLES (roof, surface)

labels = nb_df['total_result']
nb_df.drop(['total_result'], inplace=True, axis=1)
nb_df = pd.get_dummies(nb_df)
nb_df['total_result'] = labels

# ------------------------------------------------------------------------------------------------ #
## NORMALIZE DATA TO DEAL WITH NEGATIVE VALUES
# Will use MinMaxScaler() to scale all quantitative variables between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
nb_scaled = scaler.fit_transform(nb_df.iloc[:,:-1])
nb_scaled_df = pd.DataFrame(nb_scaled, columns=nb_df.iloc[:,:-1].columns)
nb_scaled_df['total_result'] = nb_df['total_result']

# --------------------------------------------------------------------------------------- #
## WRITE PREPPED DATA TO A .CSV FILE

nb_df.to_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Naive_Bayes/prepped_data/gbg_nb.csv',
              index=False)

nb_scaled_df.to_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Naive_Bayes/prepped_data/gbg_nb_scaled.csv',
              index=False)