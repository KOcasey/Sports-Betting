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
library(slam)

# ---------------------------------------------------------------------------- #
## LOAD DATA
## Must be numeric and unlabeled

# Keep row names for labeling in the dendrogram if labels exist already
My_articles_m <- as.matrix(read.csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Clustering/prepped_data/articles_prepped_for_cluster.csv', row.names = 1))
nrow(My_articles_m)


# ---------------------------------------------------------------------------- #
## Hierarchical Clustering of .txt Articles

## COsine Sim will be the distance metric used due to high dimensions
## a * b / (||a|| * ||b||)


CosineSim <- My_articles_m / sqrt(rowSums(My_articles_m * My_articles_m))
CosineSim <- CosineSim %*% t(CosineSim)

#Convert to distance metric

D_Cos_Sim <- as.dist(1-CosineSim)

# Clustering using hclust()
HClust_Ward_CosSim_SmallCorp2 <- hclust(D_Cos_Sim, method="ward.D2")

# Plots a dendrogram of the clustering results
plot(HClust_Ward_CosSim_SmallCorp2, cex=.7, hang=-11,main = "Articles")
# Plots rectangles to separate clusters for the specified number of clusters 'k'
rect.hclust(HClust_Ward_CosSim_SmallCorp2, k=3)
