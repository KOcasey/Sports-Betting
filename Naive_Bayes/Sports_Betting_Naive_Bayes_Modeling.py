# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 14:36:25 2023

@author: casey
"""

## LOAD LIBRARIES
# Set seed for reproducibility
import random; 
random.seed(53)
import pandas as pd
import numpy as np

# Import all we need from sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV # RandomizedSearchCV coming soon
from sklearn.model_selection import KFold, cross_val_score

# Import Bayesian Optimization
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval # coming soon

# Import visualization
import scikitplot as skplt
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------ #
## LOAD DATA
nb_df = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Naive_Bayes/prepped_data/gbg_nb_scaled.csv')

# ------------------------------------------------------------------------------------------------ #
## CREATE TRAIN AND TEST SETS

# X will contain all variables except the labels (the labels are the last column 'total_result')
X = nb_df.iloc[:,:-1]
# y will contain the labels (the labels are the last column 'total_result')
y = nb_df.iloc[:,-1:]

# split the data vectors randomly into 80% train and 20% test
# X_train contains the quantitative variables for the training set
# X_test contains the quantitative variables for the testing set
# y_train contains the labels for training set
# y_test contains the lables for the testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ------------------------------------------------------------------------------------------------ #
## CREATE DEFAULT MULTINOMIAL NAIVE BAYES MODEL
# default smoothing parameter alpha=1
# Look at below documentation for parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
MultiNB_Classifier = MultinomialNB()
MultiNB_Classifier.fit(X_train, y_train)

# ------------------------------------------------------------------------------------------------ #
## EVALUATE DEFAULT MULTINOMIAL MODEL
y_pred = MultiNB_Classifier.predict(X_test)

# For auc_roc
y_proba = MultiNB_Classifier.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Roc_Auc_Score: ' + str(roc_auc_score(y_test, y_proba[:, 1])))

# ------------------------------------------------------------------------------------------------ #
## GRIDSEARCHCV HYPERPARAMETER DEFAULT MULTINOMIAL MODEL
estimator = MultinomialNB()
parameters = {
    'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 4, 5]
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

# best params: alpha = 4

# ------------------------------------------------------------------------------------------------ #
## CREATE TUNED MULTINOMIAL NAIVE BAYES MODEL
MultiNB_Classifier = MultinomialNB(alpha = 4)
MultiNB_Classifier.fit(X_train, y_train)

# ------------------------------------------------------------------------------------------------ #
## EVALUATE TUNED MULTINOMIAL MODEL
y_pred = MultiNB_Classifier.predict(X_test)

# For auc_roc
y_proba = MultiNB_Classifier.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Roc_Auc_Score: ' + str(roc_auc_score(y_test, y_proba[:, 1])))

# ------------------------------------------------------------------------------------------------ #
## KFOLD CROSS VALIDATE TUNED MULTINOMIAL MODEL

MultiNB_Classifier = MultinomialNB(alpha = 4)

kf = KFold(n_splits=10, shuffle=True, random_state=1)

cv_score = cross_val_score(MultiNB_Classifier,
                           X_train, y_train, 
                           cv=kf, 
                           scoring='accuracy')

fold = 1
for score in cv_score:
    print('Fold ' + str(fold) + ' : ' + str(round(score, 2)))
    fold += 1
    
print('The mean accuracy over 10 folds is: ' + str(cv_score.mean()))

# ------------------------------------------------------------------------------------------------ #
## CREATE DEFAULT BERNOULLI NAIVE BAYES MODEL
# default smoothing parameter alpha=1
# Look at below documentation for parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html
BernNB_Classifier = BernoulliNB()
BernNB_Classifier.fit(X_train, y_train)

# ------------------------------------------------------------------------------------------------ #
## EVALUATE DEFAULT BERNOULLI MODEL
y_pred = BernNB_Classifier.predict(X_test)

# For auc_roc
y_proba = BernNB_Classifier.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Roc_Auc_Score: ' + str(roc_auc_score(y_test, y_proba[:, 1])))

# ------------------------------------------------------------------------------------------------ #
## GRIDSEARCHCV HYPERPARAMETER DEFAULT BERNOULLI MODEL
estimator = BernoulliNB()
parameters = {
    'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 4, 5]
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

# best params: alpha = 4

# ------------------------------------------------------------------------------------------------ #
## CREATE TUNED BERNOULLI NAIVE BAYES MODEL
BernNB_Classifier = BernoulliNB(alpha=4)
BernNB_Classifier.fit(X_train, y_train)

# ------------------------------------------------------------------------------------------------ #
## EVALUATE TUNED BERNOULLI MODEL
y_pred = BernNB_Classifier.predict(X_test)

# For auc_roc
y_proba = BernNB_Classifier.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Roc_Auc_Score: ' + str(roc_auc_score(y_test, y_proba[:, 1])))

# ------------------------------------------------------------------------------------------------ #
## KFOLD CROSS VALIDATE TUNED BERNOULLI MODEL

BernNB_Classifier = BernoulliNB(alpha=4)

kf = KFold(n_splits=10, shuffle=True, random_state=1)

cv_score = cross_val_score(BernNB_Classifier,
                           X_train, y_train, 
                           cv=kf, 
                           scoring='accuracy')

fold = 1
for score in cv_score:
    print('Fold ' + str(fold) + ' : ' + str(round(score, 2)))
    fold += 1
    
print('The mean accuracy over 10 folds is: ' + str(cv_score.mean()))

# ------------------------------------------------------------------------------------------------ #
## PLOT CONFUSION MATRIX

fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(121)
skplt.metrics.plot_confusion_matrix(y_test, y_pred,
                                    title="Confusion Matrix for Tuned Bernoulli Naive Bayes Model",
                                    cmap="Oranges",
                                    ax=ax1)


