# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:16:55 2023

@author: casey
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

## LOAD DATA
pokemon = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Clustering/prepped_data/Pokemon.csv')
pokemon = pokemon[['type1', 'total', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]
pokemon = pokemon[(pokemon['type1'] == 'Electric') | (pokemon['type1'] == 'Grass') | (pokemon['type1'] == 'Psychic')]

pokemon['type1'].value_counts()

## TRAIN AND TEST SETS
X = pokemon.iloc[:,1:]
y = pokemon.iloc[:,:1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

## CREATE TREE
DT_Classifier = DecisionTreeClassifier(criterion='entropy', max_depth=4)
DT_Classifier.fit(X_train, y_train)

## EVALUATE TREE
y_pred = DT_Classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

## VISUALIZE TREE
from sklearn import tree
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(DT_Classifier, 
                   feature_names=X.columns,  
                   class_names=['0','1', '2'],
                   filled=True)

fig.savefig('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Decision_Trees/visualizations/pokemon_tree.pdf')
