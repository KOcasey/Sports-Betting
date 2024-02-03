# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:54:49 2023

@author: casey
"""

# import necessary libraries
import pandas as pd

# read in game dataset
gbg = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/data/gbg_cleaned.csv')

# need to only select numeric columns
# can't use labels or categorical or text for clustering
# will keep the 'total_result' column (contains labels) for now 
#  keep to check accuracy of clustering (make sure to remove the 'total_result' before clustering)
gbg_cl = gbg[['total_result','temp', 'wind','total_line', 'away_rest', 'home_rest', 'avg_away_total_yards', 'avg_home_total_yards', 
           'avg_away_total_yards_against', 'avg_home_total_yards_against',
           'away_total_offense_rank', 'home_total_offense_rank', 
           'away_total_defense_rank', 'home_total_defense_rank',
           'elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'qbelo1_pre',
           'qbelo2_pre', 'qb1_value_pre', 'qb2_value_pre', 'qb1_adj','qb2_adj',
           'total_team_elo', 'total_qb_elo', 'team_elo_diff', 'qb_elo_diff']]


gbg_cl.to_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Clustering/prepped_data/gbg_prepped_for_cluster.csv',
              index=False)


