#########################################
##
##  Comprehensive Clustering Tutorial
##  
##  Corpus - Text - Small
##  Corpus - Text - Novels
##  
##  Record - 3D - Small
## 
##
##  k means, hclust,  and Vis
##
##  Elbow, Silhouette
##
##  Prof. Ami Gates
#####################################################


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

#############################################################################
##
##
##                  Clustering Text Data from a Corpus
##
##############################################################################
##
## Let's start small first and use a small corpus so we can see
## and understand the results.
## Then, we will cluster novels to see if they cluster by writer...
##############################################################################



## Next, load in the documents ( from the corpus)
SmallCorpus <- Corpus(DirSource("text_clust_practice"))
(getTransformations()) ## These work with library tm
(ndocs<-length(SmallCorpus)) # number of documents in corpus

## Do some clean-up.............
SmallCorpus <- tm_map(SmallCorpus, content_transformer(tolower))
SmallCorpus <- tm_map(SmallCorpus, removePunctuation)

# Remove apostrophes and quotes
removeSpecialChars <- function(x) gsub("[^a-zA-Z0-9 ]","",x)
SmallCorpus <- tm_map(SmallCorpus, removeSpecialChars)

## Remove all Stop Words
SmallCorpus <- tm_map(SmallCorpus, removeWords, stopwords("english"))

## You can also remove words that you do not want
## update after looking at wordcloud
MyStopWords <- c("and","like", "very", "can", "I", "also", "lot", 'say', 'get')
SmallCorpus <- tm_map(SmallCorpus, removeWords, MyStopWords)
SmallCorpus <- tm_map(SmallCorpus, lemmatize_strings)

##-------------------------------------------------------------

## Convert to Document Term Matrix  and TERM document matrix
## Each has its own purpose.

## DOCUMENT Term Matrix  (Docs are rows)
SmallCorpus_DTM <- DocumentTermMatrix(SmallCorpus,
                                      control = list(
                                        stopwords = TRUE, ## remove normal stopwords
                                        wordLengths=c(3, 10), ## get rid of words of len 2 or smaller or larger than 15
                                        removePunctuation = TRUE,
                                        removeNumbers = TRUE,
                                        tolower=TRUE
                                        #stemming = TRUE,
                                      ))

inspect(SmallCorpus_DTM)

## TERM Document Matrix  (words are rows)
SmallCorpus_TERM_DM <- TermDocumentMatrix(SmallCorpus,
                                          control = list(
                                            stopwords = TRUE, ## remove normal stopwords
                                            wordLengths=c(3, 10), ## get rid of words of len 2 or smaller or larger than 15
                                            removePunctuation = TRUE,
                                            removeNumbers = TRUE,
                                            tolower=TRUE
                                            #stemming = TRUE,
                                          ))

inspect(SmallCorpus_TERM_DM)

#######################

## Formats matter!
##
###-----------------------
## Convert to DF 
##------------------------
(inspect(SmallCorpus_DTM))
SmallCorpus_DF_DT <- as.data.frame(as.matrix(SmallCorpus_DTM))

(inspect(SmallCorpus_TERM_DM))
SmallCorpus_DF_TermDoc <- as.data.frame(as.matrix(SmallCorpus_TERM_DM))

############ Data frames are useful in R
SmallCorpus_DF_DT$money  ## Num of times "money" occurs in each of the 9 docs

##--------------------

## COnvert to matrix 
## -----------------------

# change number of rows to match number of documents
# in this case there are 9 documents
SC_DTM_mat <- as.matrix(SmallCorpus_DTM)
(SC_DTM_mat[1:6,1:10])

# change number of cols to match number of documents
# in this case there are 9 documents
SC_TERM_Doc_mat <- as.matrix(SmallCorpus_TERM_DM)
(SC_TERM_Doc_mat[1:10,1:6])

## WORDCLOUD ##_---------------------------------------
word.freq <- sort(rowSums(SC_TERM_Doc_mat), decreasing = T)
wordcloud(words = names(word.freq), freq = word.freq*2, min.freq = 2,
          random.order = F)

## -----------------------------

## Get Frequencies and sums
## -----------------------------------
(SmallCorpusWordFreq <- colSums(SC_DTM_mat))
## Order and sum..
(head(SmallCorpusWordFreq))
(length(SmallCorpusWordFreq))
ord <- order(SmallCorpusWordFreq)
(SmallCorpusWordFreq[head(ord)]) ## least frequent
(SmallCorpusWordFreq[tail(ord)])  ## most frequent
## Row Sums
(Row_Sum_Per_doc <- rowSums((SC_DTM_mat)))  ## total words in each row (doc)

#### Create your own normalization function to divide 
#### the frequency of each word in each row
#### by the sum of the words in that row.

SC_DTM_mat_norm <- apply(SC_DTM_mat, 1, function(i) round(i/sum(i),2))
(SC_DTM_mat_norm[1:9,1:5])

####################################################
## We have many formats of our data.
## We have a normalized DTM: SC_DTM_mat_norm
## We have data frames: SmallCorpus_DF_DT   and SmallCorpus_DF_TermDoc
## We have the Term Doc Matrix...SC_TERM_Doc_mat 
#####################################################################

###
#k-MEANS CLUSTERING

fviz_nbclust(SmallCorpus_DF_DT, method = "silhouette", 
             FUN = hcut, k.max = 5)

## k means.............on documents............
## transpose the matrix to look at similarity between documents
SC_DTM_mat_norm_t<-t(SC_DTM_mat_norm)
kmeans_smallcorp_Result <- kmeans(SC_DTM_mat_norm_t, 2, nstart=25)   

# Print the results
print(kmeans_smallcorp_Result)

kmeans_smallcorp_Result$centers  

## Place results in a table with the original data
cbind(SmallCorpus_DF_DT, cluster = kmeans_smallcorp_Result$cluster)

## See each cluster
kmeans_smallcorp_Result$cluster

## This is the size (the number of points in) each cluster
# Cluster size
kmeans_smallcorp_Result$size
## Here we have two clusters, each with 5 points (rows/vectors) 

## Visualize the clusters
fviz_cluster(kmeans_smallcorp_Result, SmallCorpus_DF_DT, 
             main="Euclidean", repel = TRUE)


## k means.............on words............
#kmeans_smallcorp_Result <- kmeans(SC_DTM_mat_norm, 6, nstart=25) 
kmeans_smallcorp_Result <- kmeans(t(SmallCorpus_DF_DT), 5, nstart=4) 

# Print the results
print(kmeans_smallcorp_Result)

kmeans_smallcorp_Result$centers  

## Place results in a table with the original data
cbind(t(SmallCorpus_DF_DT), cluster = kmeans_smallcorp_Result$cluster)

## See each cluster
kmeans_smallcorp_Result$cluster

## This is the size (the number of points in) each cluster
# Cluster size
kmeans_smallcorp_Result$size
## Here we have two clusters, each with 5 points (rows/vectors) 

## Visualize the clusters
fviz_cluster(kmeans_smallcorp_Result, t(SmallCorpus_DF_DT), 
             main="Euclidean",repel = TRUE)


##########################################################
## Let's see if we can do better with Kmeans (not kmeans)
## and different distance metrics...
#########################################################
## k = 2
My_Kmeans_SmallCorp2<-Kmeans(SmallCorpus_DF_DT, centers=2 ,method = "euclidean")
fviz_cluster(My_Kmeans_SmallCorp2, SmallCorpus_DF_DT, main="Euclidean k=2",repel = TRUE)

## k = 2
My_Kmeans_SmallCorp3<-Kmeans(t(SmallCorpus_DF_DT), centers=2 ,method = "spearman")
fviz_cluster(My_Kmeans_SmallCorp3, t(SmallCorpus_DF_DT), main="Spearman", repel = TRUE)

## k = 3
My_Kmeans_SmallCorp4<-Kmeans(t(SmallCorpus_DF_DT), centers=3 ,method = "spearman")
fviz_cluster(My_Kmeans_SmallCorp4, t(SmallCorpus_DF_DT), main="Spearman", repel = TRUE)

## k = 2 and different metric
My_Kmeans_SmallCorp4<-Kmeans(t(SmallCorpus_DF_DT), centers=2 ,method = "manhattan")
fviz_cluster(My_Kmeans_SmallCorp4, t(SmallCorpus_DF_DT), main="manhattan", repel = TRUE)

## k = 2 and different metric
My_Kmeans_SmallCorp5<-Kmeans(t(SmallCorpus_DF_DT), centers=2 ,method = "canberra")
fviz_cluster(My_Kmeans_SmallCorp5, t(SmallCorpus_DF_DT), main="canberra", repel = TRUE)

####################### Cluster the Docs and not the words with Kmeans....
## change the t (undo the transpose)
My_Kmeans_SmallCorpD<-Kmeans(SmallCorpus_DF_DT, centers=2 ,
                             method = "euclidean")
My_Kmeans_SmallCorpD$cluster
#https://www.rdocumentation.org/packages/factoextra/versions/1.0.7/topics/fviz_cluster
fviz_cluster(My_Kmeans_SmallCorpD, SmallCorpus_DF_DT, 
             main="Euclidean k = 2",repel = TRUE) +
  scale_color_brewer('Cluster', palette='Set2') + 
  scale_fill_brewer('Cluster', palette='Set2') 
#scale_shape_manual('Cluster', values=c(100,2,24, 1)


###-----------------------------------------------------------------------------
############################## Let's look at hierarchical clustering............
###-----------------------------------------------------------------------------

## Example:
(Dist_CorpusM2<- dist(SmallCorpus_DF_DT, method = "minkowski", p=2)) #Euclidean
## Now run hclust...you may use many methods - Ward, Ward.D2, complete, etc..
## see above
(HClust_SmallCorp <- hclust(Dist_CorpusM2, method = "ward.D" ))
plot(HClust_SmallCorp, cex=0.9, hang=-1, main = "Minkowski p=2 (Euclidean)")
rect.hclust(HClust_SmallCorp, k=4)

##############################################################
## Using Cosine Similarity with Ward.D2..............................
## 
## SPECIAL EXAMPLE that uses distance()  rather than dist()
##--------------------------------------------------------------------
## The distance method offer 46 metrics but requires some attention to 
## data types... notice the as.dist below...
##MORE:
## https://cran.r-project.org/web/packages/philentropy/vignettes/Distances.html
###----------------------------------------------------------------
## SCALE data before using cosine sim!
## - - - - - - 
(dist_C_smallCorp <- distance(as.matrix(scale(t(SmallCorpus_DF_DT))), method="cosine",use.row.names = TRUE))
dist_C_smallCorp<- as.dist(dist_C_smallCorp)
HClust_Ward_CosSim_SmallCorp <- hclust(dist_C_smallCorp, method="ward.D")
plot(HClust_Ward_CosSim_SmallCorp, cex=.7, hang=-1,main = "Cosine Sim")
rect.hclust(HClust_Ward_CosSim_SmallCorp, k=3)


### OR ### Create your own Cosine Sim Distance Metric function
### This one works much better...........
(My_m <- (as.matrix(scale(t(SmallCorpus_DF_DT)))))
(My_cosine_dist = 1-crossprod(My_m) /(sqrt(colSums(My_m^2)%*%t(colSums(My_m^2)))))
# create dist object
My_cosine_dist <- as.dist(My_cosine_dist) ## Important
HClust_Ward_CosSim_SmallCorp2 <- hclust(My_cosine_dist, method="ward.D")
plot(HClust_Ward_CosSim_SmallCorp2, cex=.7, hang=-30,main = "Cosine Sim")
rect.hclust(HClust_Ward_CosSim_SmallCorp2, k=3)
