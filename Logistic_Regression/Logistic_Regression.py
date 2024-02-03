# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:10:35 2023

@author: casey
"""

## LOAD LIBRARIES
# Set seed for reproducibility
import random; 
random.seed(53)
import pandas as pd

# Import all we need from sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV # RandomizedSearchCV coming soon
from sklearn.model_selection import KFold, cross_val_score

# Import Bayesian Optimization
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval # coming soon

# Import visualization
import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------------- #
## LOAD DATA

lr_df = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Logistic_Regression/prepped_data/gbg_lr_py.csv')

# --------------------------------------------------------------------------------------- #
## CREATE TRAINING AND TESTING SETS

# X will contain all variables except the labels (the labels are the last column 'total_result')
X = lr_df.iloc[:,:-1]
# y will contain the labels (the labels are the last column 'total_result')
y = lr_df.iloc[:,-1:]

# split the data vectors randomly into 80% train and 20% test
# X_train contains the quantitative variables for the training set
# X_test contains the quantitative variables for the testing set
# y_train contains the labels for training set
# y_test contains the lables for the testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# ------------------------------------------------------------------------------------------------ #
## CREATE DEFAULT LOGISTIC MODEL
# Look at below documentation for parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
LOGR_Classifier = linear_model.LogisticRegression()
LOGR_Classifier.fit(X_train, y_train)

# --------------------------------------------------------------------------------------- #
## EVALUATE MODEL

y_pred = LOGR_Classifier.predict(X_test)

# For roc_auc
y_proba = LOGR_Classifier.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_proba[:, 1]))

# ------------------------------------------------------------------------------------------------ #
## GRIDSEARCHCV HYPERPARAMETER TUNING
estimator = linear_model.LogisticRegression()
parameters = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 0.5, 1, 1.5, 5],
    'solver': ['liblinear'],
    'max_iter': [100, 500, 1000]
    }

# Use Kfold because even distribution of labels (48.5% Over, 51.5% Under)
kf = KFold(n_splits=5, shuffle=True, random_state=1)

grid_search = GridSearchCV(
    estimator = estimator,
    param_grid = parameters,
    scoring = 'accuracy',
    cv = kf,
    verbose=1
)

grid_search.fit(X_train, y_train)

# gets best params
grid_search.best_params_

# best params: C=0.5, penalty='l2', solver='liblinear', max_iter=500

# ------------------------------------------------------------------------------------------------ #
## CREATE TUNED LOGISTIC MODEL
LOGR_Classifier = linear_model.LogisticRegression(C=0.5, penalty='l2', solver='liblinear', max_iter=500)
LOGR_Classifier.fit(X_train, y_train)

# ------------------------------------------------------------------------------------------------ #
## EVALUATE TUNED MODEL
y_pred = LOGR_Classifier.predict(X_test)

# For auc_roc
y_proba = LOGR_Classifier.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Roc_Auc_Score: ' + str(roc_auc_score(y_test, y_proba[:, 1])))

# ------------------------------------------------------------------------------------------------ #
## KFOLD CROSS VALIDATE TUNED LOGISTIC MODEL

LOGR_Classifier = linear_model.LogisticRegression(C=0.5, penalty='l2', solver='liblinear', max_iter=500)

kf = KFold(n_splits=10, shuffle=True, random_state=1)

cv_score = cross_val_score(LOGR_Classifier,
                           X_train, y_train, 
                           cv=kf, 
                           scoring='accuracy')

fold = 1
for score in cv_score:
    print('Fold ' + str(fold) + ' : ' + str(round(score, 2)))
    fold += 1
    
print('The mean accuracy over 10 folds is: ' + str(cv_score.mean()))

# ------------------------------------------------------------------------------------------------ #
## CREATE REDUCED LOGISTIC MODEL (Only important features)

# only keep important features in train and test sets
X_train = X_train[['wind', 'avg_home_total_yards', 'total_line', 'qb_elo_diff', 'avg_away_total_yards', 'surface_dessograss']]
X_test = X_test[['wind', 'avg_home_total_yards', 'total_line', 'qb_elo_diff', 'avg_away_total_yards', 'surface_dessograss']]

# Look at below documentation for parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
LOGR_Classifier = linear_model.LogisticRegression()

## EVALUATE REDUCED MODEL
kf = KFold(n_splits=10, shuffle=True, random_state=1)

cv_score = cross_val_score(LOGR_Classifier,
                           X_train, y_train, 
                           cv=kf, 
                           scoring='accuracy')

fold = 1
for score in cv_score:
    print('Fold ' + str(fold) + ' : ' + str(round(score, 2)))
    fold += 1
    
print('The mean accuracy over 10 folds is: ' + str(cv_score.mean()))


# ------------------------------------------------------------------------------------------------ #
## VISUALIZATIONS

## GET FEATURE IMPORTANCE
LOGR_Classifier.coef_[0]

feat_dict= {}
for col, val in sorted(zip(X_train.columns, LOGR_Classifier.coef_[0]),key=lambda x:x[1],reverse=True):
  feat_dict[col]=val
  
feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})

## PLOT FEATURE IMPORTANCE
values = feat_df.Importance    
idx = feat_df.Feature
plt.figure(figsize=(10,8))
clrs = ['green' if (x > 0) else 'red' for x in values ]
sns.barplot(y=idx,x=values,palette=clrs).set(title='Important Features to Predict the Total Result')
plt.show()

## PLOT CONFUSION MATRIX

fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(121)
skplt.metrics.plot_confusion_matrix(y_test, y_pred,
                                    title="Confusion Matrix for Reduced Logistic Regression Model",
                                    cmap = 'Oranges',
                                    ax=ax1)

