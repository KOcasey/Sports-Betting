# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:23:49 2023

@author: casey
"""

import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_samples, silhouette_score

 
# -------------------------------------------------------------------------------------- #
## LOAD IN DATA
articles = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Clustering/prepped_data/articles_prepped_for_cluster.csv')
articles.head(5)

## Need to remove labels (can skip if data has no labels)
# keep labels in a separate list
true_labels = articles['Unnamed: 0']
print(true_labels)

# Need to rename the labels
for i in range(0, len(true_labels)):
    if i < 20:
        true_labels[i] = 'Business'
    elif i < 40:
        true_labels[i] = 'Politics'
    else:
        true_labels[i] = 'Sports Betting'
    
print(true_labels)

# remove labels
articles.drop('Unnamed: 0', inplace=True, axis=1)

# -------------------------------------------------------------------------------------- #
## K-MEANS CLUSTERING

## Create K-Means with 'K' of 2, 3, and 4
Kmeans_object2 = KMeans(n_clusters=2)
Kmeans_object3 = KMeans(n_clusters=3)
Kmeans_object4 = KMeans(n_clusters=4)

## Fit the K-Means to the dataset
Kmeans_2 = Kmeans_object2.fit(articles)
Kmeans_3 = Kmeans_object3.fit(articles)
Kmeans_4 = Kmeans_object4.fit(articles)

## Get cluster assignment labels
labels_2 = Kmeans_2.labels_
labels_3 = Kmeans_3.labels_
labels_4 = Kmeans_4.labels_

prediction_Kmeans_2 = Kmeans_object2.predict(articles)
prediction_Kmeans_3 = Kmeans_object3.predict(articles)
prediction_Kmeans_4 = Kmeans_object4.predict(articles)


## Prints out predicted labels from K-Means followed by the actual labels
print("Prediction K=2\n")
print(prediction_Kmeans_2)
print("Actual\n")
print(true_labels)

print("Prediction K=3\n")
print(prediction_Kmeans_3)
print("Actual\n")
print(true_labels)

print("Prediction K=4\n")
print(prediction_Kmeans_4)
print("Actual\n")
print(true_labels)


## Convert True Labels from text to numeric labels...
data_classes = ["Business", "Politics", "Sports Betting"]
dc = dict(zip(data_classes, range(0,3)))
print(dc)
true_label_num=true_labels.map(dc, na_action='ignore')
print(true_label_num)

## Get clustering accuracies by checking against the actual labels
acc2 = prediction_Kmeans_2 == true_label_num
acc3 = prediction_Kmeans_3 == true_label_num
acc4 = prediction_Kmeans_4 == true_label_num

print(acc2.sum() / len(acc2))
print(acc3.sum() / len(acc3))
print(acc4.sum() / len(acc4))

# ------------------------------------------------------------------------------------ #
## SILHOUETTE SCORING (find optimal K value)
import seaborn as sns

## The Silhouette Method helps to determine the optimal number of clusters
    ## in kmeans clustering...
    
    #Silhouette Coefficient = (x-y)/ max(x,y)

    #where, y is the mean intra cluster distance - the mean distance 
    ## to the other instances in the same cluster. 
    ## x depicts mean nearest cluster distance i.e. the mean distance 
    ## to the instances of the next closest cluster.
    ## The coefficient varies between -1 and 1. 
    ## A value close to 1 implies that the instance is close to its 
    ## cluster is a part of the right cluster. 
    ## Whereas, a value close to -1 means that the value is 
    ## assigned to the wrong cluster.
    
## Gets Silhouette Scores for different values of K    
s2 = silhouette_score(articles, prediction_Kmeans_2)
s3 = silhouette_score(articles, prediction_Kmeans_3)
s4 = silhouette_score(articles, prediction_Kmeans_4)

# Plots the Silhouette Scores to find optimal value of K (largest value)
x = [1, 2, 3, 4]
y = [0, s2, s3, s4]

fig = sns.lineplot(x = x, y = y)
fig.set(xlabel='K value', ylabel='Silhouette Score', 
        title='Silhouette Scores for Different Values of K')
plt.show()

# ------------------------------------------------------------------------------------- #
## LOOKING AT DISTANCES

articles.head()

## Let's find the distances between each PAIR
## of vectors. What is a vector? It is a data row.
## For example:  [84       250         17]
## Where, in this case, 84 is the value for height
## 250 is weight, and 17 is age.

X = articles

from sklearn.metrics.pairwise import euclidean_distances
## Distance between each pair of rows (vectors)
Euc_dist=euclidean_distances(X, X)

from sklearn.metrics.pairwise import manhattan_distances
Man_dist=manhattan_distances(X,X)

from sklearn.metrics.pairwise import cosine_distances
Cos_dist=cosine_distances(X,X)

from sklearn.metrics.pairwise import cosine_similarity
Cos_Sim=cosine_similarity(X,X)

#The cosine distance is equivalent to the half the squared 
## euclidean distance if each sample is normalized to unit norm

# ---------------------------------------------------------------------------------- #
## VISUALIZE DISTANCES

from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

print(Euc_dist)
X=articles
#sns.set()  #back to defaults
sns.set(font_scale=3)
Z = linkage(squareform(np.around(euclidean_distances(X), 3)))

fig4 = plt.figure(figsize=(15, 15))
ax4 = fig4.add_subplot(111)
dendrogram(Z, ax=ax4)
ax4.tick_params(axis='x', which='major', labelsize=15)
ax4.tick_params(axis='y', which='major', labelsize=15)
#ax5 = fig4.add_subplot(212)
fig4.savefig('exampleSave.png')


# ----------------------------------------------------------------------------------- #
## NORMALIZING...via scaling MIN MAX for a HEATMAP

## For the heatmap, we must normalize first
from sklearn import preprocessing

x = X.values #returns a numpy array
print(x)
#Instantiate the min-max scaler
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
articles_scaled = pd.DataFrame(x_scaled)
print(articles.columns)
sns.clustermap(articles,yticklabels=true_labels, 
               xticklabels=articles.columns)


