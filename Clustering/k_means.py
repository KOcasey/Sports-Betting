# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:01:54 2023

@author: casey
"""

import nltk
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_samples, silhouette_score

 
# -------------------------------------------------------------------------------------- #
## LOAD IN DATA
gbg_clust = pd.read_csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Clustering/prepped_data/gbg_prepped_for_cluster.csv')

## Need to remove labels (can skip if data has no labels)
# keep labels in a separate list
true_labels = gbg_clust['total_result']
print(true_labels)

# remove labels
gbg_clust.drop('total_result', inplace=True, axis=1)

# -------------------------------------------------------------------------------------- #
## K-MEANS CLUSTERING

## Create K-Means with 'K' of 2, 3, and 4
Kmeans_object2 = KMeans(n_clusters=2)
Kmeans_object3 = KMeans(n_clusters=3)
Kmeans_object4 = KMeans(n_clusters=4)

## Fit the K-Means to the dataset
Kmeans_2 = Kmeans_object2.fit(gbg_clust)
Kmeans_3 = Kmeans_object3.fit(gbg_clust)
Kmeans_4 = Kmeans_object4.fit(gbg_clust)

## Get cluster assignment labels
labels_2 = Kmeans_2.labels_
labels_3 = Kmeans_3.labels_
labels_4 = Kmeans_4.labels_

prediction_Kmeans_2 = Kmeans_object2.predict(gbg_clust)
prediction_Kmeans_3 = Kmeans_object3.predict(gbg_clust)
prediction_Kmeans_4 = Kmeans_object4.predict(gbg_clust)

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
data_classes = ["Over", "Under"]
dc = dict(zip(data_classes, range(0,2)))
print(dc)
true_label_num=true_labels.map(dc, na_action='ignore')
print(true_label_num)

# ------------------------------------------------------------------------------------- #
## LOOKING AT DISTANCES

gbg_clust.head()

## Let's find the distances between each PAIR
## of vectors. What is a vector? It is a data row.
## For example:  [84       250         17]
## Where, in this case, 84 is the value for height
## 250 is weight, and 17 is age.

X = gbg_clust

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
import seaborn as sns

print(Euc_dist)
X=gbg_clust
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
gbg_clust_scaled = pd.DataFrame(x_scaled)
print(gbg_clust.columns)
sns.clustermap(gbg_clust_scaled,yticklabels=true_labels, 
               xticklabels=gbg_clust.columns)

# ------------------------------------------------------------------------------------ #
## Silhouette and Elbow - Optimal Clusters...

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
s2 = silhouette_score(gbg_clust, prediction_Kmeans_2)
s3 = silhouette_score(gbg_clust, prediction_Kmeans_3)
s4 = silhouette_score(gbg_clust, prediction_Kmeans_4)

# Plots the Silhouette Scores to find optimal value of K (largest value)
x = [1, 2, 3, 4]
y = [0, s2, s3, s4]

fig = sns.lineplot(x = x, y = y)
fig.set(xlabel='K value', ylabel='Silhouette Score', 
        title='Silhouette Scores for different values of K')
plt.show()


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
#length of the document: called cosine similarity
cosdist = 1 - cosine_similarity(MyDTM)
print(cosdist)
print(np.round(cosdist,3))  #cos dist should be .02
