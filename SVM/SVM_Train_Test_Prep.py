# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:40:28 2023

@author: casey
"""

## LOAD LIBRARIES

import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------- #
## LOAD DATA

svm_df = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/SVM/prepped_data/gbg_svm_py.csv')

# --------------------------------------------------------------------------------------- #
## CREATE TRAINING AND TESTING SETS

# X will contain all variables except the labels (the labels are the last column 'total_result')
X = svm_df.iloc[:,:-1]
# y will contain the labels (the labels are the last column 'total_result')
y = svm_df.iloc[:,-1:]

# split the data vectors randomly into 80% train and 20% test
# X_train contains the quantitative variables for the training set
# X_test contains the quantitative variables for the testing set
# y_train contains the labels for training set
# y_test contains the lables for the testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)