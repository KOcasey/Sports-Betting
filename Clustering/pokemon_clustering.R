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
library(dplyr)

# ---------------------------------------------------------------------------- #
## LOAD DATA
## Must be numeric and unlabeled

# Keep row names for labeling in the visualizations if labels exist already
pokemon <- read.csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Clustering/prepped_data/Pokemon.csv', row.names = 2)
pokemon <- pokemon %>% 
  dplyr::select(-c(number, type1, type2, generation, legendary))
stage1 <- c('Charmander', 'Squirtle', 'Bulbasaur', 'Caterpie', 'Pidgey', 'Ekans',
            'Sandshrew', 'Vulpix', 'Zubat', 'Diglett')
stage2 <- c('Ivysaur', 'Charmeleon', 'Wartortle', 'Pidgeotto', 'Raticate', 'Arbok',
            'Nidorina', 'Gloom','Machoke', 'Graveler')
stage3 <- c('Blastoise', 'Charizard', 'Venusaur', 'Machamp', 'Golem', 'Pidgeot',
            'Vileplume', 'Swampert', 'Meganium', 'Typhlosion')
mega <- c('Mega Charizard X', 'Mega Charizard Y',
          'Mega Venusaur', 'Mega Blastoise', 'Mega Mewtwo X', 'Mega Pidgeot',
          'Mega Alakazam', 'Mega Slowbro', 'Mega Gengar', 'Mega Kangaskhan')
pokemon <- pokemon[c(stage1, mega),]

# Normalize

process <- preProcess(pokemon, method=c("range"))

pokemon_scale <- predict(process, pokemon)

# --------------------------------------------------------------------------- #
# FINDING OPTIMAL VALUE OF K
## for the FUN, use 'kmeans' for kmeans cluster or 'hcut' for hierarchical clustering

## SILHOUETTE METHOD
fviz_nbclust(pokemon_scale, FUN = hcut, method = "silhouette", k.max = 5)

## ELBOW METHOD
fviz_nbclust(pokemon_scale, FUN = kmeans, method = 'wss', k.max = 5)

## GAP-STATISTIC METHOD
fviz_nbclust(pokemon_scale, FUN = kmeans, method = 'gap_stat', k.max = 5)

# --------------------------------------------------------------------------- #
## k-MEANS CLUSTERING
kmeans_result <- kmeans(pokemon_scale, 2, nstart=25)   

# Print the results
print(kmeans_result)


kmeans_result$centers  

## Place results in a table with the original data
cbind(pokemon_scale, cluster = kmeans_result$cluster)

## See each cluster
kmeans_result$cluster

## This is the size (the number of points in) each cluster
# Cluster size
kmeans_result$size
## Here we have two clusters, each with 5 points (rows/vectors) 

## Visualize the clusters
fviz_cluster(kmeans_result, pokemon_scale, 
             main="Euclidean", repel = TRUE)

# --------------------------------------------------------------------------- #
## K-MEANS CLUSTERING

## To cluster words use transpose t()
## To cluster articles don't use transpose

## k = 2
My_Kmeans_SmallCorp2<-Kmeans(pokemon_scale, centers=2 ,method = "euclidean")
fviz_cluster(My_Kmeans_SmallCorp2, pokemon_scale, main="Euclidean k=3",repel = TRUE)

## k = 2
My_Kmeans_SmallCorp3<-Kmeans(t(pokemon_scale), centers=2 ,method = "spearman")
fviz_cluster(My_Kmeans_SmallCorp3, t(pokemon_scale), main="Spearman", repel = TRUE)

## k = 2
My_Kmeans_SmallCorp4<-Kmeans(pokemon_scale, centers=2 ,method = "spearman")
fviz_cluster(My_Kmeans_SmallCorp4, pokemon_scale, main="Spearman", repel = TRUE)

## k = 2 and different metric
My_Kmeans_SmallCorp4<-Kmeans(t(pokemon_scale), centers=2 ,method = "manhattan")
fviz_cluster(My_Kmeans_SmallCorp4, t(pokemon_scale), main="manhattan", repel = TRUE)

## k = 2 and different metric
My_Kmeans_SmallCorp5<-Kmeans(t(pokemon_scale), centers=2 ,method = "canberra")
fviz_cluster(My_Kmeans_SmallCorp5, t(pokemon_scale), main="canberra", repel = TRUE)

# ------------------------------------------------------------------------------ #
## Hierarchical Clustering of record data

## COsine Sim will be the distance metric used due to high dimensions
## a * b / (||a|| * ||b||)


CosineSim <- as.matrix(pokemon_scale) / sqrt(rowSums(as.matrix(pokemon_scale) * as.matrix(pokemon_scale)))
CosineSim <- CosineSim %*% t(CosineSim)

#Convert to distance metric

D_Cos_Sim <- as.dist(1-CosineSim)

# Clustering using hclust()
pokemon_scale_h_clust <- hclust(dist(pokemon_scale), method="ward.D2")

# Plots a dendrogram of the clustering results
plot(pokemon_scale_h_clust, cex=.7, hang=-11,main = "Pokemon")
# Plots rectangles to separate clusters for the specified number of clusters 'k'
rect.hclust(pokemon_scale_h_clust, k=2)
