#this works by using statistics but in test we do not have them
#
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


training.window <- roughtrainig[roughtrainig$new_window == "yes",]
dim(training)
qplot(training$max_roll_belt, training$max_yaw_belt, color = classe, data= training)

#keep just statistics
ind <- sapply(list("max", "min", "stddev", "avg", "amplitude", "kurtosis","skewness"),
               function(s) grep(s, names(training.window)))
ind <- sort(as.vector(ind))
data <- training.window[,ind]
#keep classes seperately
classes <- training.window$classe
#converrt all data to numeric
 for (i in 1:dim(data)[2]){
         data[,i] <- as.numeric(data[,i])
 }

#remove zero variance
indvar <- which(apply(data, 2, var) == 0)
data <- data[,-indvar]
#divide to trian and test
indtrain <- createDataPartition(classes, p = .7, list = F)
training <- data[indtrain,]
classes.training <- classes[indtrain]
testing <- data[-indtrain,]
classes.testing <- classes[-indtrain]

normalmodel <- preProcess(training, method = c("center", "scale"))
training.normal <- predict(normalmodel, training)
pcaModel <- preProcess(training.normal, method = "pca", thresh = .8)
trainingpca <- predict(pcaModel, training.normal)

#training random forest model
modelrf1<- train(trainingpca,classes.training, method="rf")
#predictin on the test data
testing.normal <- predict(normalmodel, testing)
testing.pca <- predict(pcaModel, testing.normal)
pred <- predict(modelrf1, testing.pca)
#cheching the accuracy
confusionMatrix(pred,classes.testing)
##this method of normalzing and pca is not accurate enough.
##I have to do better selecting features
#removing features which are highly correlated
correlationMatrix <- cor(data)
correlationMatrix
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=.7)
data <- data[,-highlyCorrelated]


modelTree <- train(data, classes, model="rpart")
#get variable importance
imp <- importance(modelTree$finalModel)
order <- order(importance(modelTree$finalModel))
cbind(rownames(imp)[order], imp[order])
#choose the 15 most important variables
impVars <- tail(rownames(imp)[order], 15)
DataImp <- data[,impVars]
#seting test and train again
training <- DataImp[indtrain,]
classes.testing <- classes[-indtrain]
testing <- DataImp[-indtrain,]
classes.testing <- classes[-indtrain]
#trian a RF model on important data
modelRFImp <- train(training, classes.training, method = "rf")
#checking the accuracy
pred <- predict(modelRFImp, testing)
confusionMatrix(pred,classes.testing)


#get variable importance
imp <- importance(modelTree$finalModel)
order <- order(importance(modelTree$finalModel))
cbind(rownames(imp)[order], imp[order])
#choose the 15 most important variables
impVars <- tail(rownames(imp)[order], 10)
DataImp <- data[,impVars]
#seting test and train again
training <- DataImp[indtrain,]
classes.testing <- classes[-indtrain]
testing <- DataImp[-indtrain,]
classes.testing <- classes[-indtrain]
#trian a RF model on important data
modelRFImp <- train(training, classes.training, method = "rf")
#checking the accuracy
pred <- predict(modelRFImp, testing)
confusionMatrix(pred,classes.testing)
