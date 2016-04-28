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
 for (i in 1:dim(training)[2]){
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
qplot(PC1, PC2, color=classes, data=trainingpca)


kmclus <- kmeans(trainingpca, centers = 5)
table(kmclus$cluster, classes)
#training random forest model
modelrf1<- train(trainingpca,classes.training, method="rf")
#predictin on the test data
testing.normal <- predict(normalmodel, testing)
testing.pca <- predict(pcaModel, testing.normal)
pred <- predict(modelrf1, testing.pca)
#cheching the accuracy
confusionMatrix(pred,classes.testing)



for (i in names(training)){
        plot(classes.training,training[,i], xlab = i)
        line <- readline()
}

testing.window <- roughtesting[roughtesting$new_window == "yes",]
indt <- sapply(list("max", "min", "stddev", "avg", "amplitude", "kurtosis","skewness"),
              function(s) grep(s, names(testing.window)))
indt <- sort(as.vector(indt))
testing <- testing.window[,indt]
testing.normal <- predict(normalmodel, testing)
testingpca <- predict(pcaModel, testing)

pred <- predict(modelrf1, testing )

featurePlot()
qplot(training[,maxs],training$classe, color=training$classe)
qplot(max_yaw_arm, data=cbind(training[, maxs], classe=training$classe), 
      color= classe, facets = max_yaw_arm~.)





df <- data.frame(training[, maxs], classe=training$classe)
melted <- melt(df, id.vars = c("max_yaw_arm","classe"),
               variable_name = "Max...")

qplot(max_yaw_arm, value, color = Max... ,facets=classe~., data=melted, geom = "line")
qplot(max_yaw_arm, value, facets = Max...~. ,color=classe, data=melted, geom = "line")

