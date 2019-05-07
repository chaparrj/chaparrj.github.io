
# Download data
library(data.table)
fileurl = 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
if (!file.exists('pml-training.csv')){
  download.file(fileurl,'./pml-training.csv', mode = 'wb')
  #  unzip("UCI HAR Dataset.zip", exdir = getwd())
}

fileurl = 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
if (!file.exists('pml-testing.csv')){
  download.file(fileurl,'./pml-testing.csv', mode = 'wb')
  #  unzip("UCI HAR Dataset.zip", exdir = getwd())
}

# Create initial data sets
allTrain <- read.table("pml-training.csv", sep = ",", header = TRUE, stringsAsFactors = FALSE)
# allCompleteTrain <- allTrain[complete.cases(allTrain), ]
allTest <- read.table("pml-testing.csv", sep = ",", header = TRUE, stringsAsFactors = FALSE)
# Feature selection on Train and Test data sets
# so that same data are used to train and test the algorithm.
# Test data has no variable to be predicted
myTrain <- allTrain[, grep("_[xyz]$", names(allTrain))]
myTrain$classe <- allTrain$classe
myTest <- allTest[, grep("_[xyz]$", names(allTest))]

library(caret)
library(MLmetrics)
library(dplyr)
inTrain <- createDataPartition(y=myTrain$classe, p=0.75, list=FALSE)
dsTrain <- myTrain[inTrain,]
dsCval <- myTrain[-inTrain,]
set.seed(1235)
# Coming command takes 66 secs to run
# system.time(rpModelFit <- train(classe~., 
#                 data=dsTrain, 
#                 preProcess = "pca",
#                 method="rpart",
#                  tuneLength = 100))
# Preprocessing with PCA produces no benefit; not used to KISS
# Coming command takes ca. 60 secs to run
system.time(rpFit <- train(classe~., 
                                data=dsTrain, 
                                method="rpart",
                                tuneLength = 150))

# Coming command takes 900 secs / 15 mins to run
# system.time(gbmModelFit <- train(classe~., 
#                    data=dsTrain, 
#                    preProcess = "pca",
#                    method="gbm"))
# Preprocessing with PCA produces no benefit; not used to KISS
# Coming command takes ca. 1204 secs (= 20 mins) to run
system.time(gbmFit <- train(classe~., 
                                 data=dsTrain, 
                                 method="gbm"))
# Testing here to run Logistic Regression; no success yet
# Coming command takes  to run
lrControl <- trainControl(method = "cv", 
                          number = 10,
                          savePredictions = TRUE)
system.time(glmFit <- train(classe~., data=dsTrain, 
                                     method = "glm",
                                     family = binomial(),
                                     trControl = lrControl)
            )
# Testing randomForest training.
# training the algorithm takes too long to complete
# rfModelFit <- train(classe~., data=dsTrain, 
#                   preProcess = "pca",
#                   method="rf",
#                   prox = TRUE)
print(rpFit)
print(gbmFit)

rpPred <- predict(rpFit, dsCval)
gbmPred <- predict(gbmFit, dsCval)
Accuracy(rpPred, dsCval$classe)
Accuracy(gbmPred, dsCval$classe)
confusionMatrix(rpPred, as.factor(dsCval$classe))
confusionMatrix(gbmPred, as.factor(dsCval$classe))

myPred <- data.frame(X = seq(1:20),
                     rpPred = predict(rpFit, myTest),
                     gbmPred = predict(gbmFit, myTest))
myPred <- mutate(myPred, 
                 samePred = ifelse(rpPred == gbmPred, TRUE, FALSE))
