# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:35:03 2023

@author: casey
"""

## LOAD LIBRARIES

import pandas as pd

# Import for preprocessing
from sklearn.preprocessing import LabelEncoder # for xgboost
from sklearn.preprocessing import MinMaxScaler # for logistic regression


# Import all we need from sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

# Import XGBoost
import xgboost as xgb

# Import visualization
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------------- #
## LOAD DATA

df = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/data/gbg_cleaned.csv')

# --------------------------------------------------------------------------------------- #
## PREPROCESSING
# --------------------------------------------------------------------------------------- #


# --------------------------------------------------------------------------------------- #
## SELECT COLUMNS FOR  MODELING

cols = ['total_result', 'total_line', 'temp', 'wind', 'total_qb_elo', 'team_elo_diff',
        'qb_elo_diff', 'avg_home_total_yards', 'avg_away_total_yards', 
        'avg_home_total_yards_against', 'avg_away_total_yards_against', 'roof', 'surface',
        'div_game']

dt_rf_df = df[cols] # df for decision trees and random forest
xg_df = df[cols] # df for xgboost
lr_df= df[cols] # df for logistic regression

# --------------------------------------------------------------------------------------- #
## CONVERT CATEGORICAL LABELS TO NUMERIC LABELS FOR XGBOOST

le = LabelEncoder()
xg_df['total_result'] = le.fit_transform(xg_df['total_result'])

# --------------------------------------------------------------------------------------- #
## ONE HOT ENCODING OF QUALITATIVE CATEGORICAL VARIABLES (roof, surface)
## FOR DECISION TREE/RANDOM FOREST, XGBOOST, AND LOGISTIC REGRESSION

labels = dt_rf_df['total_result']
dt_rf_df.drop(['total_result'], inplace=True, axis=1)
dt_rf_df = pd.get_dummies(dt_rf_df)
dt_rf_df['total_result'] = labels

labels = xg_df['total_result']
xg_df.drop(['total_result'], inplace=True, axis=1)
xg_df = pd.get_dummies(xg_df)
xg_df['total_result'] = labels

labels = lr_df['total_result']
lr_df.drop(['total_result'], inplace=True, axis=1)
lr_df = pd.get_dummies(lr_df)
lr_df['total_result'] = labels

# --------------------------------------------------------------------------------------- #
## NORMALIZE DATA USING MINMAXSCALER FOR LOGISTIC REGRESSION ONLY
# Will use MinMaxScaler() to scale all quantitative variables between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
lr_df_scaled = scaler.fit_transform(lr_df.iloc[:,:-1])
lr_df_scaled = pd.DataFrame(lr_df_scaled, columns=lr_df.iloc[:,:-1].columns)
lr_df_scaled['total_result'] = lr_df['total_result']

# --------------------------------------------------------------------------------------- #
## MODELING
# --------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------ #
## CREATE TRAIN AND TEST SETS FOR DECISION TREE/RANDOM FOREST

# X will contain all variables except the labels (the labels are the last column 'total_result')
X = dt_rf_df.iloc[:,:-1]
# y will contain the labels (the labels are the last column 'total_result)
y = dt_rf_df.iloc[:,-1:]

# split the data vectors randomly into 80% train and 20% test
# X_train contains the quantitative variables for the training set
# X_test contains the quantitative variables for the testing set
# y_train contains the labels for training set
# y_test contains the lables for the testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CREATE FULL DECISION TREE MODEL
# Look at below documentation for parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
DT_Classifier = DecisionTreeClassifier(criterion='entropy', max_depth=7)
DT_Classifier.fit(X_train, y_train)

# CREATE FULL RANDOM FOREST MODEL
# Look at below documentation for parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
RF_Classifier = RandomForestClassifier(criterion='entropy')
RF_Classifier.fit(X_train, y_train)


# ------------------------------------------------------------------------------------------------ #
## CREATE TRAIN AND TEST SETS FOR XGBOOST

# X will contain all variables except the labels (the labels are the last column 'total_result')
X = xg_df.iloc[:,:-1]
# y will contain the labels (the labels are the last column 'total_result)
y = xg_df.iloc[:,-1:]

# split the data vectors randomly into 80% train and 20% test
# X_train contains the quantitative variables for the training set
# X_test contains the quantitative variables for the testing set
# y_train contains the labels for training set
# y_test contains the lables for the testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CREATE DEFAULT XGBOOST MODEL
# documentation for parameters
# https://xgboost.readthedocs.io/en/latest/parameter.html
xgb_Classifier = xgb.XGBClassifier()
xgb_Classifier.fit(X_train, y_train)

# ------------------------------------------------------------------------------------------------ #
## CREATE TRAIN AND TEST SETS FOR LOGISTIC REGRESSION

# X will contain all variables except the labels (the labels are the last column 'total_result')
X = lr_df_scaled.iloc[:,:-1]
# y will contain the labels (the labels are the last column 'total_result)
y = lr_df_scaled.iloc[:,-1:]

# split the data vectors randomly into 80% train and 20% test
# X_train contains the quantitative variables for the training set
# X_test contains the quantitative variables for the testing set
# y_train contains the labels for training set
# y_test contains the lables for the testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CREATE DEFAULT LOGISTIC REGRESSION MODEL
# Look at below documentation for parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
LOGR_Classifier = linear_model.LogisticRegression()
LOGR_Classifier.fit(X_train, y_train)


# --------------------------------------------------------------------------------------- #
## FEATURE IMPORTANCE
# --------------------------------------------------------------------------------------- #

## GET FEATURE IMPORTANCE FROM DECISION TREE
feat_dict= {}
for col, val in sorted(zip(X_train.columns, DT_Classifier.feature_importances_),key=lambda x:x[1],reverse=True):
  feat_dict[col]=val
  
feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})

## PLOT FEATURE IMPORTANCE FROM DECISION TREE
values = feat_df.Importance    
idx = feat_df.Feature
plt.figure(figsize=(10,8))
clrs = ['green' if (x < max(values)) else 'red' for x in values ]
sns.barplot(y=idx,x=values,palette=clrs).set(title='Important Features to Predict NFL Game Total Line Results (Decision Tree)')
plt.show()

## GET FEATURE IMPORTANCE FROM RANDOM FOREST
feat_dict= {}
for col, val in sorted(zip(X_train.columns, RF_Classifier.feature_importances_),key=lambda x:x[1],reverse=True):
  feat_dict[col]=val
  
feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})

## PLOT FEATURE IMPORTANCE FROM RANDOM FOREST
values = feat_df.Importance    
idx = feat_df.Feature
plt.figure(figsize=(10,8))
clrs = ['green' if (x < max(values)) else 'red' for x in values ]
sns.barplot(y=idx,x=values,palette=clrs).set(title='Important Features to Predict NFL Game Total Line Results (Random Forest)')
plt.show()

## GET FEATURE IMPORTANCE FROM XGBOOST
feat_dict= {}
for col, val in sorted(zip(X_train.columns, xgb_Classifier.feature_importances_),key=lambda x:x[1],reverse=True):
  feat_dict[col]=val
  
feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})

## PLOT FEATURE IMPORTANCE FROM XGBOOST
values = feat_df.Importance    
idx = feat_df.Feature
plt.figure(figsize=(10,8))
clrs = ['green' if (x < max(values)) else 'red' for x in values ]
sns.barplot(y=idx,x=values,palette=clrs).set(title='Important Features to Predict NFL Game Total Line Results (XGBoost)')
plt.show()

# GET FEATURE IMPORTANCE FROM LOGISTIC REGRESSION
LOGR_Classifier.coef_[0]

feat_dict= {}
for col, val in sorted(zip(X_train.columns, LOGR_Classifier.coef_[0]),key=lambda x:x[1],reverse=True):
  feat_dict[col]=val
  
feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})

# PLOT FEATURE IMPORTANCE FROM LOGISTIC REGRESSION
values = feat_df.Importance    
idx = feat_df.Feature
plt.figure(figsize=(10,8))
clrs = ['green' if (x > 0) else 'red' for x in values ]
sns.barplot(y=idx,x=values,palette=clrs).set(title='Important Features to Predict NFL Game Total Line Results (Logistic Regression)')
plt.show()

# --------------------------------------------------------------------------------------- #
## FEATURE SELECTION
# --------------------------------------------------------------------------------------- #

## GET FEATURE SELECTION FROM XGBOOST
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

 
X = xg_df.iloc[:,:-1]
# y will contain the labels (the labels are the last column 'total_result)
y = xg_df.iloc[:,-1:]
# split the data vectors randomly into 80% train and 20% test
# X_train contains the quantitative variables for the training set
# X_test contains the quantitative variables for the testing set
# y_train contains the labels for training set
# y_test contains the lables for the testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CREATE DEFAULT XGBOOST MODEL
# documentation for parameters
# https://xgboost.readthedocs.io/en/latest/parameter.html
xgb_Classifier = xgb.XGBClassifier()
xgb_Classifier.fit(X_train, y_train)

# make predictions for test data and evaluate
predictions = xgb_Classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(xgb_Classifier.feature_importances_)
accuracies = []
for thresh in thresholds:
 # select features using threshold
 selection = SelectFromModel(xgb_Classifier, threshold=thresh, prefit=True)
 select_X_train = selection.transform(X_train)
 # train model
 selection_model = XGBClassifier()
 selection_model.fit(select_X_train, y_train)
 # eval model
 select_X_test = selection.transform(X_test)
 predictions = selection_model.predict(select_X_test)
 accuracy = accuracy_score(y_test, predictions)
 accuracies.append("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
 
 for accuracy in accuracies:
     print(accuracy)