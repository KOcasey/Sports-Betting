## LOAD LIBRARIES

library(dplyr)
library(rpart)   ## FOR Decision Trees
library(rattle)  ## FOR Decision Tree Vis
library(rpart.plot)
library(RColorBrewer)
library(Cairo)
library(network)
library(ggplot2)
library(slam)
library(quanteda)
library(proxy)
library(igraph)
library(caret)
library(randomForest)

# ------------------------------------------------------------------------------ #
## LOAD DATA

gbg_dt_full <- read.csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Decision_Trees/prepped_data/gbg_dt_full_r.csv')

# ------------------------------------------------------------------------------ #
## SET VARIABLES TO CORRECT DATA TYPES

gbg_dt_full$total_result <- as.factor(gbg_dt_full$total_result)
gbg_dt_full$game_type <- as.factor(gbg_dt_full$game_type)
gbg_dt_full$weekday <- as.factor(gbg_dt_full$weekday)
gbg_dt_full$location <- as.factor(gbg_dt_full$location)
gbg_dt_full$roof <- as.factor(gbg_dt_full$roof)
gbg_dt_full$surface <- as.factor(gbg_dt_full$surface)


str(gbg_dt_full)

# ------------------------------------------------------------------------------ #
## SPLIT DATA INTO TRAIN AND TEST

# will split 80% train and 20% test
# check to see how big the training and testing datasets should be after splitting the data
nrow(gbg_dt_full)*0.8
nrow(gbg_dt_full)*0.2

## set a seed if you want it to be the same each time you
## run the code. The number (like 1234) does not matter
set.seed(1234)

# find the number corresponding to 80% of the data
n <- floor(0.8*nrow(gbg_dt_full)) 

# randomly sample indicies to be included in our training set (80%)
index <- sample(seq_len(nrow(gbg_dt_full)), size = n)

# set the training set to be randomly sampled rows of the data (80%)
train <- gbg_dt_full[index, ]

# set the testing set to be the remaining rows (20%)
test <- gbg_dt_full[-index, ] 

# check to see if the size of the training and testing sets match what was expected
cat("There are", dim(train)[1], "rows and", dim(train)[2], 
    "columns in the training set.")
cat("There are", dim(test)[1], "rows and", dim(test)[2], 
    "columns in the testing set.")

# make sure the testing and training sets have balanced labels
table(train$total_result)
table(test$total_result)

# remove labels from testing set and keep them
test_labels <- test$total_result
test <- test[ , -which(names(test) %in% c("total_result"))]

write.csv(train, 'C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Decision_Trees/prepped_data/dt_train_r.csv', row.names = FALSE)
write.csv(test, 'C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Decision_Trees/prepped_data/dt_test_r.csv', row.names = FALSE)
