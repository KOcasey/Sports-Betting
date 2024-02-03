# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 11:10:55 2023

@author: casey
"""

## LOAD LIBRARIES
# Set seed for reproducibility
import random; 
random.seed(53)
import numpy as np
import pandas as pd

# Import all we need from sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV # RandomizedSearchCV coming soon
from sklearn.model_selection import KFold, cross_val_score

# Import Bayesian Optimization
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval # coming soon

# Import visualization
import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

# ------------------------------------------------------------------------------------------------ #
## LOAD DATA
dt_df = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Decision_Trees/prepped_data/gbg_dt.csv')

# ------------------------------------------------------------------------------------------------ #
## CREATE TRAIN AND TEST SETS

# X will contain all variables except the labels (the labels are the last column 'total_result')
X = dt_df.iloc[:,:-1]
# y will contain the labels (the labels are the last column 'total_result')
y = dt_df.iloc[:,-1:]

# split the data vectors randomly into 80% train and 20% test
# X_train contains the quantitative variables for the training set
# X_test contains the quantitative variables for the testing set
# y_train contains the labels for training set
# y_test contains the lables for the testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ------------------------------------------------------------------------------------------------ #
## CREATE DEFAULT TREE
# Look at below documentation for parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# I began with a depth of 7, but some leaf nodes had few samples (1, 2, 3) indicating overfitting, same with 6,
#  depth of 5 looks a lot better so that's why default start with depth 5
DT_Classifier = DecisionTreeClassifier(max_depth=5)
DT_Classifier.fit(X_train, y_train)

# ------------------------------------------------------------------------------------------------ #
## EVALUATE DEFAULT TREE (Depth 5)
y_pred = DT_Classifier.predict(X_test)

# For auc_roc
y_proba = DT_Classifier.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Roc_Auc_Score: ' + str(roc_auc_score(y_test, y_proba[:, 1])))

# ------------------------------------------------------------------------------------------------ #
## GRIDSEARCHCV HYPERPARAMETER TUNING
estimator = DecisionTreeClassifier()
parameters = {
    'max_depth': [3, 4, 5],
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': [4, 5],
    'min_samples_split': [5, 10, 20]
    }

# Use Kfold because even distribution of labels (48.5% Over, 51.5% Under)
kf = KFold(n_splits=10, shuffle=True, random_state=1)

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

# best params: max_depth=3, criterion='gini', min_samples_leaf=4, min_samples_split=10

# ------------------------------------------------------------------------------------------------ #
## CREATE TUNED TREE
DT_Classifier = DecisionTreeClassifier(max_depth=3, criterion='gini', min_samples_leaf=4, min_samples_split=10)
DT_Classifier.fit(X_train, y_train)

# ------------------------------------------------------------------------------------------------ #
## EVALUATE TUNED TREE
y_pred = DT_Classifier.predict(X_test)

# For auc_roc
y_proba = DT_Classifier.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Roc_Auc_Score: ' + str(roc_auc_score(y_test, y_proba[:, 1])))

# ------------------------------------------------------------------------------------------------ #
## KFOLD CROSS VALIDATE TUNED TREE

DT_Classifier = DecisionTreeClassifier(max_depth=3, criterion='gini', min_samples_leaf=4, min_samples_split=5)

kf = KFold(n_splits=10, shuffle=True, random_state=1)

cv_score = cross_val_score(DT_Classifier,
                           X_train, y_train, 
                           cv=kf, 
                           scoring='accuracy')

fold = 1
for score in cv_score:
    print('Fold ' + str(fold) + ' : ' + str(round(score, 2)))
    fold += 1
    
print('The mean accuracy over 10 folds is: ' + str(cv_score.mean()))

# ------------------------------------------------------------------------------------------------ #
## CREATE REDUCED TREE (Only important features)

# only keep important features in train and test sets
X_train = X_train[['wind', 'qb_elo_diff', 'avg_home_total_yards', 'total_qb_elo']]
X_test = X_test[['wind', 'qb_elo_diff', 'avg_home_total_yards', 'total_qb_elo']]

# Look at below documentation for parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
DT_Classifier = DecisionTreeClassifier(max_depth=3, criterion='gini', min_samples_leaf=4, min_samples_split=5)
DT_Classifier.fit(X_train, y_train)

## EVALUATE TREE
y_pred = DT_Classifier.predict(X_test)

# For auc_roc
y_proba = DT_Classifier.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_proba[:, 1]))

# ------------------------------------------------------------------------------------------------ #
## VISUALIZATIONS

## GET FEATURE IMPORTANCE
feat_dict= {}
for col, val in sorted(zip(X_train.columns, DT_Classifier.feature_importances_),key=lambda x:x[1],reverse=True):
  feat_dict[col]=val
  
feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})

## PLOT FEATURE IMPORTANCE
values = feat_df.Importance    
idx = feat_df.Feature
plt.figure(figsize=(10,8))
clrs = ['green' if (x < max(values)) else 'red' for x in values ]
sns.barplot(y=idx,x=values,palette=clrs).set(title='Important Features to Predict NFL Game Total Line Results')
plt.show()

## VISUALIZE TREE
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(DT_Classifier, 
                   feature_names=X.columns,  
                   class_names=['Over','Under'],
                   filled=True)

fig.savefig('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Decision_Trees/visualizations/GS_Reduced_Tree_py.pdf')

## PLOT CONFUSION MATRIX
fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(121)
skplt.metrics.plot_confusion_matrix(y_test, y_pred,
                                    title="Confusion Matrix for GridSearchCV Tuned Reduced Decision Tree",
                                    cmap="Oranges",
                                    ax=ax1)