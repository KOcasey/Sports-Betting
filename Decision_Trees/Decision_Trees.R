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


# ------------------------------------------------------------------------------ #
## DECISION TREE MODELING

## TRAIN THE MODEL WITH TRAIN SET
# this code uses rpart to create decision tree
# here, the ~ .  means to train using all data variables
# the train$total_result tells it what the label is called
# in this dataset, the label is called "total_result"
# the smaller the cp the larger the tree, if cp is too small you have overfitting
DT_1 <- rpart(total_result ~ ., data = train, method="class",
              cp=0, maxdepth=7)
summary(DT_1)

# this is the cp plot
plotcp(DT_1) 

# find which variables are most important
DT_1$variable.importance

## TEST THE MODEL WITH THE TEST SET

DT_1_pred <- predict(DT_1, test, type="class")

# ------------------------------------------------------------------------------ #
## NEW DECISION TREE WITH UNIMPORTANT VARIABLES REMOVED
## REDUCED MODEL Version 1 (Max Depth of 8)
DT_2 <- rpart(total_result ~ off_def_diff + wind + total_line + surface + 
              total_qb_elo + temp, 
              data = train, method="class",
              cp=0, maxdepth = 8)

summary(DT_2)

# find which variables are most important
DT_2$variable.importance

# this is the cp plot
plotcp(DT_2) 

# TEST THE NEW DECISION TREE
DT_2_pred <- predict(DT_2, test, type="class")

# ---------------------------------------------------------------------------- #
## NEW DECISION TREE WITH UNIMPORTANT VARIABLES REMOVED
## REDUCED MODEL Version 2 (Max Depth of 6)
DT_3 <- rpart(total_result ~ off_def_diff + wind + total_line + surface + 
                total_qb_elo + temp, 
              data = train, method="class",
              cp=0, maxdepth = 6)

summary(DT_3)

# find which variables are most important
DT_3$variable.importance

# this is the cp plot
plotcp(DT_3) 

# TEST THE NEW DECISION TREE
DT_3_pred <- predict(DT_3, test, type="class")

# ---------------------------------------------------------------------------- #
## EVALUATE RESULTS

# FULL TREE (DT_1)
# Visualization of the tree
fancyRpartPlot(DT_1)
prp(DT_1, main='Decision Tree (Full Model)')

# Confusion Matrix 
cm_1 <- confusionMatrix(DT_1_pred, test_labels)


# REDUCED TREE (DT_2)
# Visualization of the tree
fancyRpartPlot(DT_2, main='Decision Tree', cex=0.1)
prp(DT_2, main='Decision Tree (Reduced Model)')

# Confusion Matrix 
cm_2 <- confusionMatrix(DT_2_pred, test_labels)

# REDUCED TREE (DT_3)
# Visualization of the tree
fancyRpartPlot(DT_3, main='Decision Tree', cex=0.1)
prp(DT_3, main='Decision Tree (Reduced Model)')

# Confusion Matrix 
cm_3 <- confusionMatrix(DT_3_pred, test_labels)

# function to create a viz for the confusion matrix
draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Over', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Under', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Over', cex=1.2, srt=90)
  text(140, 335, 'Under', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  

draw_confusion_matrix(cm_1)
draw_confusion_matrix(cm_2)
draw_confusion_matrix(cm_3)

# ---------------------------------------------------------------------------- #
## K-Fold CROSS-VALIDATION

# select number of folds (value of K)
# select how many times to repeat
train_control <- trainControl(method = 'repeatedcv', number = 5, repeats = 2)

# keep complexity parameter constant
tune_grid = expand.grid(cp=c(0))

# perform cross-validation
cv_tree <- train(total_result ~ off_def_diff + wind + total_line + surface + 
                 total_qb_elo + temp, 
                 data = train, method="rpart",
                 trControl = train_control,
                 tuneGrid = tune_grid,
                 maxdepth = 8)

# view results of cross-validation
cv_tree

#################################################################
## Extra notes about the output/summary
## - Root Node Error x  - X Error  - is the cross-validated error rate, 
## which is a more objective measure of predictive accuracy
##  - Root Node Error x  - Rel Error -  is the resubstitution 
## error rate (the error rate computed on the training sample).

## Variable Importance: The values are calculate by summing up all the 
## improvement measures that each variable contributes
## RE: the sum of the goodness of split measures for each split 
## for which it was the primary variable

## in Summary, the variable importance sums to 100

## NOTE: variable.importance is a named numeric vector giving 
## the importance of each variable. (Only present
## if there are any splits.) 
## When printed by summary.rpart these are rescaled to
## add to 100.
###########################################################


############################################################
## READ MORE about cross-validation:
## 1) http://inferate.blogspot.com/2015/05/k-fold-cross-validation-with-decision.html
## 2) This one is mathy and complicated for those who are interested
##    https://rafalab.github.io/dsbook/cross-validation.html
## 3) I like this one...
##    https://www.kaggle.com/satishgunjal/tutorial-k-fold-cross-validation
##
