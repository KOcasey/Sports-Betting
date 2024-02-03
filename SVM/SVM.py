# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:18:52 2023

@author: casey
"""

## LOAD LIBRARIES
# Set seed for reproducibility
import random; random.seed(53)
import pandas as pd

# Import all we need from sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
from sklearn.model_selection import cross_validate

# Import visualization
import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sns

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

# --------------------------------------------------------------------------------------- #
## TRAIN SVM MODEL

# change kernel to achieve different accuracies (linear, sigmoid, rbf, poly)
# C is the regularization parameter, must be positive, default is 1.0
# for poly, need to specify degree, default is 3
# SVM_Classifier = svm.SVC(kernel='poly', degree=2)
# for a list of all parameters go the following documentation
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
model = svm.SVC(kernel='rbf', C=1.0)
svm_classifier = model.fit(X_train, y_train)

# --------------------------------------------------------------------------------------- #
## EVALUATE MODEL

y_pred = svm_classifier.predict(X_test)
svm_accuracy_score = accuracy_score(y_test, y_pred)
svm_results = classification_report(y_test, y_pred)
print('SVM Accuracy Score: ', svm_accuracy_score)
print(confusion_matrix(y_test, y_pred))
print(svm_results)

# ------------------------------------------------------------------------------------- #
## PLOT CONFUSION MATRIX

fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(121)
skplt.metrics.plot_confusion_matrix(y_pred, y_test,
                                    title="Confusion Matrix for Term Frequency Vectorizer",
                                    cmap="Oranges",
                                    ax=ax1)

