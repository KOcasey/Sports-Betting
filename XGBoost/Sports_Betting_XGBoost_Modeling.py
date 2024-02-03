# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 14:40:31 2023

@author: casey
"""

## LOAD LIBRARIES
import pandas as pd

# Import all we need from sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV # RandomizedSearchCV coming soon
from sklearn.model_selection import KFold, cross_val_score

# Import Bayesian Optimization
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval # coming soon

# Import XGBoost
import xgboost as xgb

# Import visualization
import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

# ------------------------------------------------------------------------------------------------ #
## LOAD DATA
xg_df = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/XGBoost/prepped_data/gbg_xg.csv')

# ------------------------------------------------------------------------------------------------ #
## CREATE TRAIN AND TEST SETS

# X will contain all variables except the labels (the labels are the first column 'survived')
X = xg_df.iloc[:,:-1]
# y will contain the labels (the labels are the first column 'survived')
y = xg_df.iloc[:,-1:]
               
# split the data vectors randomly into 80% train and 20% test
# X_train contains the quantitative variables for the training set
# X_test contains the quantitative variables for the testing set
# y_train contains the labels for training set
# y_test contains the lables for the testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ------------------------------------------------------------------------------------------------ #
## CREATE DEFAULT XGBOOST MODEL
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn

# documentation for parameters
# https://xgboost.readthedocs.io/en/latest/parameter.html

# default parameters
# increasing max_depth could lead to overfitting
# alpha/lambda are regularization parameters 
# gamma typically between 0 and 5, with increasing regularization
# learning rate range is 0.01-0.3
# n_estimators is number of trees range of 100-1000
xgb_Classifier = xgb.XGBClassifier()
xgb_Classifier.fit(X_train, y_train)

## EVALUATE MODEL
y_pred = xgb_Classifier.predict(X_test)

# For auc_roc
y_proba = xgb_Classifier.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_proba[:, 1]))

# ------------------------------------------------------------------------------------------------ #
## GRIDSEARCHCV HYPERPARAMETER TUNING
# took ~ 1hr 30 min

estimator = xgb.XGBClassifier()
parameters = {
    'learning_rate' : [0.01, 0.1, 0.3],
    'gamma': [0, 0.8, 2],
    'max_depth':[5, 6, 7],
    'min_child_weight':[1, 3, 5],
    'subsample':[0.5, 0.8],
    'colsample_bytree':[0.5, 0.8],
    'n_estimators':[100, 300, 500],
    'lambda':[.5, 1]
    }

# Use Kfold because even distribution of labels (48.5% Over, 51.5% Under)
kf = KFold(n_splits=3, shuffle=True, random_state=1)

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

# best params: {'colsample_bytree': 0.8, 'gamma': 0, 'lambda': 0.5, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 0.8}

# ------------------------------------------------------------------------------------------------ #
## CREATE TUNED XGBOOST
xgb_Classifier = xgb.XGBClassifier(colsample_bytree=0.8, gamma=0, reg_lambda=0.5, learning_rate=0.1, max_depth=6, min_child_weight=1, n_estimators=100, subsample=0.8)
xgb_Classifier.fit(X_train, y_train)

## EVALUATE TUNED XGBOOST
y_pred = xgb_Classifier.predict(X_test)

# For auc_roc
y_proba = xgb_Classifier.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_proba[:, 1]))

# ------------------------------------------------------------------------------------------------ #
## KFOLD CROSS VALIDATE TUNED RANDOM FOREST

xgb_Classifier = xgb.XGBClassifier(colsample_bytree=0.8, gamma=0, reg_lambda=0.5, learning_rate=0.1, max_depth=6, min_child_weight=1, n_estimators=100, subsample=0.8)

kf = KFold(n_splits=10, shuffle=True, random_state=1)

cv_score = cross_val_score(xgb_Classifier,
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

## VISUALIZE A TREE
# .estimators_[0] is the first tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(xgb_Classifier.estimators_[0], 
                   feature_names=X.columns,  
                   class_names=['0','1'],
                   filled=True)

fig.savefig('C:/Users/casey/OneDrive/Documents/Machine_Learning/Supervised_Learning/Random_Forest/Visualizations/Random_Forest_Default.pdf')


## PLOT CONFUSION MATRIX
fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(121)
skplt.metrics.plot_confusion_matrix(y_test, y_pred,
                                    title="Confusion Matrix for Tuned XGBoost",
                                    cmap="Oranges",
                                    ax=ax1)
