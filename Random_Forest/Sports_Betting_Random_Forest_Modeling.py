# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 14:46:30 2023

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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score, recall_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV # RandomizedSearchCV coming soon
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Import Bayesian Optimization
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval # coming soon

# Import visualization
import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

import pickle

import optuna

# ------------------------------------------------------------------------------------------------ #
## LOAD DATA
rf_df = pd.read_csv('C:/Users/casey/OneDrive/Documents/Data_Science/Projects/Sports-Betting/Random_Forest/prepped_data/gbg_rf.csv')
rf_df.info()

# ------------------------------------------------------------------------------------------------ #
# Encode Labels to 0 and 1
# Over:0, Under:1
le = LabelEncoder()
rf_df['total_result'] = le.fit_transform(rf_df['total_result'])

# ------------------------------------------------------------------------------------------------ #
## CREATE TRAIN AND TEST SETS

# X will contain all variables except the labels (the labels are the first column 'survived')
X = rf_df.iloc[:,:-1]
# y will contain the labels (the labels are the first column 'survived')
y = np.array(rf_df.iloc[:,-1:]).ravel()

# split the data vectors randomly into 80% train and 20% test
# X_train contains the quantitative variables for the training set
# X_test contains the quantitative variables for the testing set
# y_train contains the labels for training set
# y_test contains the lables for the testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ------------------------------------------------------------------------------------------------ #
## CREATE DEFAULT RANDOM FOREST
# Look at below documentation for parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
RF_Classifier = RandomForestClassifier()
RF_Classifier.fit(X_train, y_train)

# ------------------------------------------------------------------------------------------------ #
## EVALUATE DEFAULT RANDOM FOREST
y_pred = RF_Classifier.predict(X_test)

# For auc_roc
y_proba = RF_Classifier.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_proba[:, 1]))

# ------------------------------------------------------------------------------------------------ #
## OPTUNA HYPERPARAMETER TUNING

def objective(trial, X, y):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 200, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        'weight_0': trial.suggest_float('weight_0', 0.1, 4),
        'weight_1': trial.suggest_float('weight_1', 0.1, 4),
    }

   

    # Fit the model

    optuna_model = RandomForestClassifier(
      n_estimators=param['n_estimators'],
      max_depth=param['max_depth'],
      min_samples_split=param['min_samples_split'],
      min_samples_leaf=param['min_samples_leaf'],
      bootstrap=param['bootstrap'],
      class_weight={0:param['weight_0'], 1:param['weight_1']},
    )
    optuna_model.fit(X_train, y_train)

    # Make predictions
    y_pred = optuna_model.predict(X_test)


    return f1_score(y_test, y_pred, average='macro')

# Generate a new study resetting trials to 0
study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    direction="maximize"
)

# study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, score_function), n_trials=100)
study.optimize(lambda trial: objective(trial, X, y), n_trials=80)

print('Best Trial: ', study.best_trial)
print('Best Params: ',study.best_params)
best_params = study.best_params

# ------------------------------------------------------------------------------------------------ #
## SAVE BEST MODEL
RF_Classifier = RandomForestClassifier(n_estimators=167, max_depth=11, min_samples_split=8, min_samples_leaf=8, bootstrap=True,
                                        class_weight={0:best_params['weight_0'], 1:best_params['weight_1']}
                                        )

RF_Classifier.fit(X_train, y_train)
pickle.dump(RF_Classifier, open('C:/Users/casey/OneDrive/Documents/Data_Science/Projects/Sports-Betting/saved_models/rf_model.pkl', open='wb'))
                                             

# ------------------------------------------------------------------------------------------------ #
## GRIDSEARCHCV HYPERPARAMETER TUNING
# took ~ 1hr 30 min

estimator = RandomForestClassifier()
parameters = {
    'n_estimators': [100, 300, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 5, 6, 7, 8, 9, 10],
    'min_samples_leaf': [3, 5, 7],
    'min_samples_split': [5, 7, 9],
    'max_features': [2, 4]
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

# best params: 'criterion': 'gini', 'max_depth': 9, 'max_features': 2, 'min_samples_leaf': 5, 'min_samples_split': 9, 'n_estimators': 100

# ------------------------------------------------------------------------------------------------ #
## CREATE TUNED RANDOM FOREST
RF_Classifier = RandomForestClassifier(criterion='gini', max_depth=9, max_features=2, min_samples_leaf=5, min_samples_split=9, n_estimators=100)
RF_Classifier.fit(X_train, y_train)

# ------------------------------------------------------------------------------------------------ #
## EVALUATE TUNED RANDOM FOREST
y_pred = RF_Classifier.predict(X_test)

# For auc_roc
y_proba = RF_Classifier.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_proba[:, 1]))

# ------------------------------------------------------------------------------------------------ #
## KFOLD CROSS VALIDATE TUNED RANDOM FOREST

RF_Classifier = RandomForestClassifier(n_estimators=167, max_depth=11, min_samples_split=8, min_samples_leaf=8, bootstrap=True,
                                        class_weight={0:best_params['weight_0'], 1:best_params['weight_1']}
                                        )
                                                      

kf = KFold(n_splits=5, shuffle=True, random_state=1)

cv_score = cross_val_score(RF_Classifier,
                           X_train, y_train, 
                           cv=kf, 
                           scoring=make_scorer(f1_score, average='macro')
                           )

fold = 1
for score in cv_score:
    print('Fold ' + str(fold) + ' : ' + str(round(score, 2)))
    fold += 1
    
print('The mean accuracy over 5 folds is: ' + str(cv_score.mean()))

# ------------------------------------------------------------------------------------------------ #
## GET FEATURE IMPORTANCE
feat_dict= {}
for col, val in sorted(zip(X_train.columns, RF_Classifier.feature_importances_),key=lambda x:x[1],reverse=True):
  feat_dict[col]=val
  
feat_df = pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})

## PLOT FEATURE IMPORTANCE
values = feat_df.Importance    
idx = feat_df.Feature
plt.figure(figsize=(10,8))
clrs = ['green' if (x < max(values)) else 'red' for x in values ]
sns.barplot(y=idx,x=values,palette=clrs).set(title='Important Features to Predict NFL Game Total Line Results')
plt.show()

# ------------------------------------------------------------------------------------------------ #
## CREATE REDUCED RANDOM FOREST (Only important features)

# only keep important features in train and test sets
X_train_red = X_train[[ 'avg_away_total_yards_against', 'team_elo_diff', 'avg_home_total_yards', 'qb_elo_diff', 'avg_away_total_yards', 'avg_home_total_yards_against', 'total_qb_elo', 'total_line', 'wind', 'temp']]
X_test_red = X_test[[ 'avg_away_total_yards_against', 'team_elo_diff', 'avg_home_total_yards', 'qb_elo_diff', 'avg_away_total_yards', 'avg_home_total_yards_against', 'total_qb_elo', 'total_line', 'wind', 'temp']]

# Look at below documentation for parameters
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
RF_Classifier = RandomForestClassifier(criterion='gini', max_depth=9, max_features=2, min_samples_leaf=5, min_samples_split=9, n_estimators=100)
RF_Classifier.fit(X_train_red, y_train)

## EVALUATE REDUCED RANDOM FOREST (ONLY IMPORTANT FEATURES)
y_pred = RF_Classifier.predict(X_test_red)

# For auc_roc
y_proba = RF_Classifier.predict_proba(X_test_red)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_proba[:, 1]))

# ------------------------------------------------------------------------------------------------ #
## VISUALIZATIONS

## VISUALIZE A TREE
# .estimators_[0] is the first tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(RF_Classifier.estimators_[0], 
                   feature_names=X.columns,  
                   class_names=['0','1'],
                   filled=True)

fig.savefig('C:/Users/casey/OneDrive/Documents/Machine_Learning/Supervised_Learning/Random_Forest/Visualizations/Random_Forest_Default.pdf')


## PLOT CONFUSION MATRIX
fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(121)
skplt.metrics.plot_confusion_matrix(y_test, y_pred,
                                    title="Confusion Matrix for Tuned Random Forest",
                                    cmap="Oranges",
                                    ax=ax1)
