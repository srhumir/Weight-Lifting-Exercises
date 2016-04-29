library(caret)
library(dplyr)
library(plyr)
library(reshape)
#downloading and reading the data
if (!file.exists("pml-trainig.csv")){
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                      destfile = "pml-trainig.csv")
}
if (!file.exists("pml-testing.csv")){
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                      destfile = "pml-testing.csv")
}
roughtrainig <- read.csv("pml-trainig.csv")
roughtesting <- read.csv("pml-testing.csv")

#exploring the trainig set
dim(roughtrainig)
names(roughtrainig)[1:10]
head(roughtrainig[,1:10])

#Getting the field with statistics
indlist <- sapply(list("max", "min", "stddev", "avg", "amplitude", "kurtosis","skewness", "^var"),
              function(s) grep(s, names(roughtrainig)))
ind <-vector() 
for (i in 1:length(indlist)){
        ind <- c(ind, indlist[[i]])
}
data <- roughtrainig[,-ind]
names(data)
#removing first 7 columns
data <- data[, -c(1:7)]
#converting all data to numeric
for (i in 1:(dim(data)[2]-1)){
        data[,i] <- as.numeric(data[,i])
}

#removing features which are highly correlated
correlationMatrix <- cor(data[,-dim(data)[2]])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=.8)
data <- data[,-highlyCorrelated]

#from here we go two seperate ways. one summerizing data by pca
#and other choosing best features.
indSensorList <- sapply(c("_belt", "_arm", "_dumbbell", "_forearm"), 
                        function(s) grep(s,names(data)))
#divide to trian and test
set.seed(1234)
indtrain <- createDataPartition(data$classe, p = .5, list = F)
training <- data[indtrain,]
testing <- data[-indtrain,]
#pca on trainig
pcalistmodel <- lapply(indSensorList, function(x) 
                                preProcess(training[,x], 
                                method = "pca", thresh = .8))
trainingpcalist <- sapply(1:4, function(i) 
                        predict(pcalistmodel[[i]],training[,indSensorList[[i]]]))
trainingpca <- data.frame(classe=training$classe)

for (i in 1:length(trainingpcalist)){
        trainingpca <- cbind(trainingpca, trainingpcalist[[i]])
}
names(trainingpca)[-1] <- sapply(1:(dim(trainingpca)[2]-1),
                                 function(i) paste("PC", i, sep=""))
#train RF model
modelTreepca <- train(classe~.,data=trainingpca, method="rpart")
#predict on test data
testingpcalist <- sapply(1:4, function(i) 
        predict(pcalistmodel[[i]],testing[,indSensorList[[i]]]))
testingpca <- data.frame(classe=testing$classe)

for (i in 1:length(testingpcalist)){
        testingpca <- cbind(testingpca, testingpcalist[[i]])
}
names(testingpca)[-1] <- sapply(1:(dim(testingpca)[2]-1),
                                 function(i) paste("PC", i, sep=""))

pred <- predict(modelTreepca, testingpca)
#confusion matrix
confusionMatrix(pred, testing$classe)
#the result is awful
#reduced the data size to run RF
ImpVars <- row.names(varImp(modelTreepca)$importance)[varImp(modelTreepca)$importance >0]
trainingRFpca <- trainingpca[,ImpVars]
modelRFpca <- train(trainingRFpca, training$classe, method = "rf")

#predict on testing
testingRFpca <- testingpca[, ImpVars]
predRFpca <- predict(modelRFpca, testingRFpca)
#confusiom matrix
confusionMatrix(predRFpca, testing$classe)
##ok well 92% accuracy but not yet conviencing

#this time do not rum pca. just choosign the most important features
modelTree <- train(classe~., data=training, method="rpart")
Impvars2 <- row.names(varImp(modelTree)$importance)[varImp(modelTree)$importance >25]
trainingRF <- training[,Impvars2]
modelRF <- train(trainingRF, training$classe, method = "rf")
#test the model
testinRF <- testing[,Impvars2]
predRF <- predict(modelRF, testing)
confusionMatrix(predRF, testing$classe)
##better 96% accuracy

#run it on the rough testing
##remove statistics
validation <- roughtesting[,-ind]
#removing first 7 columns
validation <- validation[, -c(1:7)]
#converting all data to numeric
for (i in 1:(dim(validation)[2]-1)){
        validation[,i] <- as.numeric(validation[,i])
}

##removing highly coorelated
validation <- validation[,-highlyCorrelated]
validation <- validation[,Impvars2]
predVali <- predict(modelRF, validation)
data.frame(num=as.factor(1:20),predVali)
