library(stats)
# clustering libraries
library(NbClust)
library(cluster)
library(mclust)
library(amap)  ## for using Kmeans (notice the cap K)
library(factoextra) ## for cluster vis, silhouette, etc.
library(purrr)
library(stylo)  ## for dist.cosine
library(philentropy)  ## for distance() which offers 46 metrics
library(SnowballC)
library(caTools)
library(dplyr)
library(textstem)
library(stringr)
library(wordcloud)
library(tm) ## to read in corpus (text data)
library(dplyr)

# ---------------------------------------------------------------------------- #
## LOAD DATA
## Must be numeric and unlabeled

# Keep row names for labeling in the visualizations if labels exist already
articles <- read.csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Clustering/prepped_data/articles_prepped_for_cluster.csv', row.names = 1)
nrow(articles)

# --------------------------------------------------------------------------- #
# FINDING OPTIMAL VALUE OF K
## for the FUN, use 'kmeans' for kmeans cluster or 'hcut' for hierarchical clustering

## SILHOUETTE METHOD
fviz_nbclust(articles, FUN = kmeans, method = "silhouette", k.max = 5)

## ELBOW METHOD
fviz_nbclust(articles, FUN = kmeans, method = 'wss', k.max = 5)

## GAP-STATISTIC METHOD
fviz_nbclust(articles, FUN = kmeans, method = 'gap_stat', k.max = 5)

# --------------------------------------------------------------------------- #
## k-MEANS CLUSTERING
kmeans_result <- kmeans(articles, 2, nstart=25)   

# Print the results
print(kmeans_result)


kmeans_result$centers  

## Place results in a table with the original data
cbind(articles, cluster = kmeans_result$cluster)

## See each cluster
kmeans_result$cluster

## This is the size (the number of points in) each cluster
# Cluster size
kmeans_result$size
## Here we have two clusters, each with 5 points (rows/vectors) 

## Visualize the clusters
fviz_cluster(kmeans_result, articles, 
             main="Euclidean", repel = TRUE)

# --------------------------------------------------------------------------- #
## K-MEANS CLUSTERING

## To cluster words use transpose t()
## To cluster articles don't use transpose

## k = 2
My_Kmeans_SmallCorp2<-Kmeans(articles, centers=2 ,method = "euclidean")
fviz_cluster(My_Kmeans_SmallCorp2, articles, main="Euclidean k=3",repel = TRUE)

## k = 2
My_Kmeans_SmallCorp3<-Kmeans(t(articles), centers=2 ,method = "spearman")
fviz_cluster(My_Kmeans_SmallCorp3, t(articles), main="Spearman", repel = TRUE)

## k = 2
My_Kmeans_SmallCorp4<-Kmeans(articles, centers=2 ,method = "spearman")
fviz_cluster(My_Kmeans_SmallCorp4, articles, main="Spearman", repel = TRUE)

## k = 2 and different metric
My_Kmeans_SmallCorp4<-Kmeans(t(articles), centers=2 ,method = "manhattan")
fviz_cluster(My_Kmeans_SmallCorp4, t(aricles), main="manhattan", repel = TRUE)

## k = 2 and different metric
My_Kmeans_SmallCorp5<-Kmeans(t(SmallCorpus_DF_DT), centers=2 ,method = "canberra")
fviz_cluster(My_Kmeans_SmallCorp5, t(articles), main="canberra", repel = TRUE)


