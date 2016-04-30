Introduction
============

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement - a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, our goal is
to use data from accelerometers on the belt, forearm, arm, and dumbbell
of six participants. They were asked to perform barbell lifts correctly
and incorrectly in five different ways. One totally right and the four
others with a common mistake. We will predict how well a given person
done the job using these accelerometers.

More information is available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

Synopsis
========

The provided dataset contains the the name of volunteer, time stamp, 56
measurement by accelometers, 96 statistics calculated by moving window
on them and the class of activity, "A" for totally right and "B-E" for
four common mistakes. As we are supposed to predict the class of
activity based on single measurements by acceleometers, we can not use
statistics. So we are limited to 56 measurements. These are still too
many variables and make the training too slow. To reduce the variables.
We first omit the highly correlated variables. Then train a
classification tree on the remaining data. Finally choose the most
important variables based on this model. Finally train a random forest
model on the remaining variables. The final out of sample accuracy based
on 50% of data for training and 50% for testing is 98.5%. We did not use
cross validation as there are enough data (around 20 thousand) in the
training set.

Details of doing the job
========================

### Loading the dataset

First of all the data are downloaded and loaded into R

    library(caret)
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

### Exploring and cleaning the dataset

Then I took a look at variables names.

    print(names(roughtrainig))

    ##   [1] "X"                        "user_name"               
    ##   [3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
    ##   [5] "cvtd_timestamp"           "new_window"              
    ##   [7] "num_window"               "roll_belt"               
    ##   [9] "pitch_belt"               "yaw_belt"                
    ##  [11] "total_accel_belt"         "kurtosis_roll_belt"      
    ##  [13] "kurtosis_picth_belt"      "kurtosis_yaw_belt"       
    ##  [15] "skewness_roll_belt"       "skewness_roll_belt.1"    
    ##  [17] "skewness_yaw_belt"        "max_roll_belt"           
    ##  [19] "max_picth_belt"           "max_yaw_belt"            
    ##  [21] "min_roll_belt"            "min_pitch_belt"          
    ##  [23] "min_yaw_belt"             "amplitude_roll_belt"     
    ##  [25] "amplitude_pitch_belt"     "amplitude_yaw_belt"      
    ##  [27] "var_total_accel_belt"     "avg_roll_belt"           
    ##  [29] "stddev_roll_belt"         "var_roll_belt"           
    ##  [31] "avg_pitch_belt"           "stddev_pitch_belt"       
    ##  [33] "var_pitch_belt"           "avg_yaw_belt"            
    ##  [35] "stddev_yaw_belt"          "var_yaw_belt"            
    ##  [37] "gyros_belt_x"             "gyros_belt_y"            
    ##  [39] "gyros_belt_z"             "accel_belt_x"            
    ##  [41] "accel_belt_y"             "accel_belt_z"            
    ##  [43] "magnet_belt_x"            "magnet_belt_y"           
    ##  [45] "magnet_belt_z"            "roll_arm"                
    ##  [47] "pitch_arm"                "yaw_arm"                 
    ##  [49] "total_accel_arm"          "var_accel_arm"           
    ##  [51] "avg_roll_arm"             "stddev_roll_arm"         
    ##  [53] "var_roll_arm"             "avg_pitch_arm"           
    ##  [55] "stddev_pitch_arm"         "var_pitch_arm"           
    ##  [57] "avg_yaw_arm"              "stddev_yaw_arm"          
    ##  [59] "var_yaw_arm"              "gyros_arm_x"             
    ##  [61] "gyros_arm_y"              "gyros_arm_z"             
    ##  [63] "accel_arm_x"              "accel_arm_y"             
    ##  [65] "accel_arm_z"              "magnet_arm_x"            
    ##  [67] "magnet_arm_y"             "magnet_arm_z"            
    ##  [69] "kurtosis_roll_arm"        "kurtosis_picth_arm"      
    ##  [71] "kurtosis_yaw_arm"         "skewness_roll_arm"       
    ##  [73] "skewness_pitch_arm"       "skewness_yaw_arm"        
    ##  [75] "max_roll_arm"             "max_picth_arm"           
    ##  [77] "max_yaw_arm"              "min_roll_arm"            
    ##  [79] "min_pitch_arm"            "min_yaw_arm"             
    ##  [81] "amplitude_roll_arm"       "amplitude_pitch_arm"     
    ##  [83] "amplitude_yaw_arm"        "roll_dumbbell"           
    ##  [85] "pitch_dumbbell"           "yaw_dumbbell"            
    ##  [87] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
    ##  [89] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"  
    ##  [91] "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   
    ##  [93] "max_roll_dumbbell"        "max_picth_dumbbell"      
    ##  [95] "max_yaw_dumbbell"         "min_roll_dumbbell"       
    ##  [97] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
    ##  [99] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
    ## [101] "amplitude_yaw_dumbbell"   "total_accel_dumbbell"    
    ## [103] "var_accel_dumbbell"       "avg_roll_dumbbell"       
    ## [105] "stddev_roll_dumbbell"     "var_roll_dumbbell"       
    ## [107] "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
    ## [109] "var_pitch_dumbbell"       "avg_yaw_dumbbell"        
    ## [111] "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
    ## [113] "gyros_dumbbell_x"         "gyros_dumbbell_y"        
    ## [115] "gyros_dumbbell_z"         "accel_dumbbell_x"        
    ## [117] "accel_dumbbell_y"         "accel_dumbbell_z"        
    ## [119] "magnet_dumbbell_x"        "magnet_dumbbell_y"       
    ## [121] "magnet_dumbbell_z"        "roll_forearm"            
    ## [123] "pitch_forearm"            "yaw_forearm"             
    ## [125] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"  
    ## [127] "kurtosis_yaw_forearm"     "skewness_roll_forearm"   
    ## [129] "skewness_pitch_forearm"   "skewness_yaw_forearm"    
    ## [131] "max_roll_forearm"         "max_picth_forearm"       
    ## [133] "max_yaw_forearm"          "min_roll_forearm"        
    ## [135] "min_pitch_forearm"        "min_yaw_forearm"         
    ## [137] "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
    ## [139] "amplitude_yaw_forearm"    "total_accel_forearm"     
    ## [141] "var_accel_forearm"        "avg_roll_forearm"        
    ## [143] "stddev_roll_forearm"      "var_roll_forearm"        
    ## [145] "avg_pitch_forearm"        "stddev_pitch_forearm"    
    ## [147] "var_pitch_forearm"        "avg_yaw_forearm"         
    ## [149] "stddev_yaw_forearm"       "var_yaw_forearm"         
    ## [151] "gyros_forearm_x"          "gyros_forearm_y"         
    ## [153] "gyros_forearm_z"          "accel_forearm_x"         
    ## [155] "accel_forearm_y"          "accel_forearm_z"         
    ## [157] "magnet_forearm_x"         "magnet_forearm_y"        
    ## [159] "magnet_forearm_z"         "classe"

The names of variables containing statistics started with one of the
words, "max", "min", "stddev", "avg", "amplitude", "kurtosis","skewness"
or "var". So I checked which columns contain these data and removed
them.

    #Getting the variables containig statistics
    indlist <- sapply(list("max", "min", "stddev", "avg", "amplitude", "kurtosis","skewness", "^var"),
                  function(s) grep(s, names(roughtrainig)))
    ind <-vector() 
    for (i in 1:length(indlist)){
            ind <- c(ind, indlist[[i]])
    }
    data <- roughtrainig[,-ind]

### Cleaning the dataset

Then I removed the first seven columns which do not actually contain
measurements.

    #removing first 7 columns
    data <- data[, -c(1:7)]

The variables need to be converted to numeric. As they might be saved as
character.

    #converting all data to numeric
    for (i in 1:(dim(data)[2]-1)){
            data[,i] <- as.numeric(data[,i])
    }

### Summarizing the data

From this step I start summarizing the data. First I remove variables
which are highly correlated to others. "caret" package has tow very
useful functions for this, "cor" which produce a correlation matrix. and
"findCorrelation" which cut too correlated variables.

    #removing features which are highly correlated
    correlationMatrix <- cor(data[,-dim(data)[2]])
    highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=.8)
    data <- data[,-highlyCorrelated]

13 variables are removed leaving 39 variables. But it is still too much
for running a random forest. First divide the dataset to training and
testing subsets half of data to each subset.

    #divide to trian and test
    set.seed(1234)
    indtrain <- createDataPartition(data$classe, p = .5, list = F)
    training <- data[indtrain,]
    testing <- data[-indtrain,]

To reduce the number of variables, one idea would be using PCA. I tried
that as well but did not get better results with lots of more processing
time. So I decided to use another method.

In the other approach I used the variable importance function in
"caret". I firstly trained a decision tree model (rpart) on all the data
which by the way did not show a good out of sample accuracy.

    modelTree <- train(classe~., data=training, method="rpart")

Then ran the varImp to get the most important variables.

    print(varImp(modelTree))

    ## rpart variable importance
    ## 
    ##   only 20 most important variables shown (out of 39)
    ## 
    ##                   Overall
    ## magnet_dumbbell_y  100.00
    ## magnet_belt_y       89.00
    ## total_accel_belt    80.95
    ## yaw_belt            74.74
    ## pitch_forearm       68.55
    ## magnet_dumbbell_z   52.17
    ## magnet_arm_x        31.78
    ## gyros_belt_z        27.88
    ## roll_forearm        23.79
    ## roll_dumbbell       21.83
    ## magnet_dumbbell_x   21.59
    ## accel_dumbbell_y    16.89
    ## roll_arm            15.58
    ## accel_forearm_x     14.50
    ## magnet_forearm_z    12.47
    ## accel_forearm_y      0.00
    ## magnet_belt_x        0.00
    ## yaw_dumbbell         0.00
    ## accel_forearm_z      0.00
    ## accel_arm_y          0.00

I filtered the variables which have 14% or more importance.

    Impvars2 <- row.names(varImp(modelTree)$importance)[varImp(modelTree)$importance >14]

Leave us by 14 variables.

### Trainig the model

Now I separated just these 14 variables and train a random forest model
on them.

    trainingRF <- training[,Impvars2]
    modelRF <- train(trainingRF, training$classe, method = "rf")

### Testing the model

Now it is time to test the model. First choose important variables on
the testing dataset.

    #test the model
    testinRF <- testing[,Impvars2]

Then predict the classes using our model

    predRF <- predict(modelRF, testing)

Finally make the confusion matrix.

    confMat <- confusionMatrix(predRF, testing$classe)

The overall accuracy of the model is 98.52% acceptingly good. The better
news is that the sensitivity and specificity on class "A" (doing the job
right) are respectively 0.9975 and 0.9974. This means that when the
prediction is "A" we are 99.75% confident that the job was done right
and when the prediction is not "A", we are 99.74% confident that the job
was really done wrong.

Predicting the provided test data
=================================

In this section I run the model on the test dataset provided and loaded
in first section by name "roughtesting".

First the dataset should look like our training dataset i.e. extra
variables have to be removed and variables have to be converted to
numeric.

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
    ##removng less important variables
    validation <- validation[,Impvars2]

Now predict the class of each row and show the result

    predVali <- predict(modelRF, validation)
    print(data.frame(Row.Number=as.factor(1:20),Predicted.Class=predVali))

    ##    Row.Number Predicted.Class
    ## 1           1               B
    ## 2           2               A
    ## 3           3               B
    ## 4           4               A
    ## 5           5               A
    ## 6           6               C
    ## 7           7               D
    ## 8           8               B
    ## 9           9               A
    ## 10         10               A
    ## 11         11               B
    ## 12         12               C
    ## 13         13               B
    ## 14         14               A
    ## 15         15               E
    ## 16         16               E
    ## 17         17               A
    ## 18         18               B
    ## 19         19               B
    ## 20         20               B

Reference
=========

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H.
Qualitative Activity Recognition of Weight Lifting Exercises.
Proceedings of 4th International Conference in Cooperation with SIGCHI
(Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Read more: <http://groupware.les.inf.puc-rio.br/har#ixzz47LMYDnT4>
