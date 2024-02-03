## The following code is courtesy of 
## Professor Ami Gates, Dept. Applied Math, Data Science, University of Colorado
## and has been adapted to fit this project

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
## LOAD .txt documents from the given folder (corpus) containing the .txt documents
ArticlesCorpus <- Corpus(DirSource("sp_po_bu_articles_20"))
(getTransformations()) ## These work with library tm

# gets num of documents in the folder
(ndocs<-length(ArticlesCorpus))


# ---------------------------------------------------------------------------- #
## TEXT DATA CLEANING for multiple .txt documents
## Do some clean-up.............

# Convert all words to lowercase
ArticlesCorpus <- tm_map(ArticlesCorpus, content_transformer(tolower))

# Remove any punctuation
ArticlesCorpus <- tm_map(ArticlesCorpus, removePunctuation)

# Creat a list of stop words to remove
MyStopWords <- c("and","like", "very", "can", "I", "also", "lot", 'say', 'get',
                 'said', 'will')

# Remove stop words from the above list
ArticlesCorpus <- tm_map(ArticlesCorpus, removeWords, MyStopWords)

##-------------------------------------------------------------

## Convert to Document Term Matrix 
## If clustering by Word is desired then change DocumentTermMatrix() to TermDocumentMatrix()

## DOCUMENT Term Matrix  (Docs are rows)
# Removes normal stopwords as well
# Only keeps word lengths between 4 and 10
# Removes punctutation
# Removes numbers
# Converts all words to lowercase
ArticlesCorpus_DTM <- DocumentTermMatrix(ArticlesCorpus,
                                       control = list(
                                         stopwords = TRUE, ## remove normal stopwords
                                         wordLengths=c(4, 10), 
                                         removePunctuation = TRUE,
                                         removeNumbers = TRUE,
                                         tolower=TRUE
                                         #stemming = TRUE,
                                       ))

#inspect(ArticlesCorpus_DTM)

# Scale and Normalize
#ArticlesCorpus_DTM <- ArticlesCorpus_DTM / row_sums(ArticlesCorpus_DTM)

#ArticlesCorpus_DTM <- weightTfIdf(ArticlesCorpus_DTM, normalize=TRUE)

### Create a DF as well................
Articles_DF_DT <- as.data.frame(as.matrix(ArticlesCorpus_DTM))

## Create .csv file of cleaned and prepped articles
# Need to use write.csv instead of write_csv to keep row names
write.csv(Articles_DF_DT, 'C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Clustering/prepped_data/articles_prepped_for_cluster.csv')

## Create a Articles Matrix for a Word Cloud
My_articles_m <- (as.matrix(Articles_DF_DT))
nrow(My_articles_m)

## WORD CLOUD
# Creates a word cloud for the articles
word.freq <- sort(rowSums(t(My_articles_m)), decreasing = T)
wordcloud(words = names(word.freq), freq = word.freq*2, min.freq = 2,
  random.order = F)
