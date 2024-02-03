## LOAD LIBRARIES

library(dplyr)
library(ggplot2)
library(naivebayes)
library(tidyverse)
library(caret)
library(caretEnsemble)
library(psych)
library(Amelia)
library(mice)
library(GGally)
library(e1071)
library(klaR)

# ------------------------------------------------------------------------------ #
## LOAD DATA

pokemon <- read.csv('C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/Clustering/prepped_data/Pokemon.csv')
pokemon <- pokemon %>% dplyr::select(-c(number, type2, legendary, generation, name))
pokemon <- pokemon %>% filter(type1 == 'Electric' | type1 == 'Rock')

# ------------------------------------------------------------------------------ #
## SET VARIABLES TO CORRECT DATA TYPES

pokemon$type1 <- as.factor(pokemon$type1)

str(pokemon)

# ------------------------------------------------------------------------------ #
## SPLIT DATA INTO TRAIN AND TEST

# will split 80% train and 20% test
# check to see how big the training and testing datasets should be after splitting the data
nrow(pokemon)*0.8
nrow(pokemon)*0.2

## set a seed if you want it to be the same each time you
## run the code. The number (like 1234) does not matter
#set.seed(1234)

# find the number corresponding to 80% of the data
n <- floor(0.8*nrow(pokemon)) 

# randomly sample indicies to be included in our training set (80%)
index <- sample(seq_len(nrow(pokemon)), size = n)

# set the training set to be randomly sampled rows of the data (80%)
train <- pokemon[index, ]

# set the testing set to be the remaining rows (20%)
test <- pokemon[-index, ] 

# check to see if the size of the training and testing sets match what was expected
cat("There are", dim(train)[1], "rows and", dim(train)[2], 
    "columns in the training set.")
cat("There are", dim(test)[1], "rows and", dim(test)[2], 
    "columns in the testing set.")

# make sure the testing and training sets have balanced labels
table(train$type1)
table(test$type1)


# remove labels from training and testing set and keep them
test_labels <- test$type1
train_labels <- train$type1
test <- test[ , -which(names(test) %in% c("type1"))]
train <- train[ , -which(names(train) %in% c("type1"))]

# ------------------------------------------------------------------------------ #
## NAIVE BAYES MODELING

NB_1_e1071_train <- naiveBayes(train, train_labels, laplace = 1)
NB_1_e1071_pred <- predict(NB_1_e1071_train, test)

# print probabilities
#NB_1_e1071_train
# ------------------------------------------------------------------------------- #
## EVALUATE RESULTS

cm_1 <- confusionMatrix(NB_1_e1071_pred, test_labels)
cm_1

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

# ------------------------------------------------------------------------------ #
## CROSS-VALIDATION

# K-Fold CROSS-VALIDATION

# select number of folds (value of K)
# select how many times to repeat
train_control <- trainControl(method = 'cv', number = 10)

# perform cross-validation
cv_nb <- train(x = train, y = train_labels, method="nb",
               trControl = train_control)

# view results of cross-validation
cv_nb

cv_nb_cm <- confusionMatrix(cv_nb)
cv_nb_cm
