#=======================================================================================
#   JH-CaseStudy-Main.R
#
#   Johns-Hopkins - Practical Machine Learning - Case Study
#
#   You should create a report describing how you built your model, 
#   how you used cross validation, 
#   what you think the expected out of sample error is, 
#   and why you made the choices you did.
#
#     Mike Pastor  January 12, 2022
 
rm( list = ls() )    # Clear Environment objects

#  Load the Packages
#
#  install.packages( "crosstable" )

library(caret)
library(dplyr)
library(ggplot2)
library(stringr)
library( lubridate )
library( crosstable )

library(AppliedPredictiveModeling)
library(ElemStatLearn)
library(rpart)        # Recursive partition trees
library(randomForest) # Random forest
library(gbm)          # Boosting

library( mgcv )       # Stacking
library( nlme )       # Stacking

set.seed( 123456 )    # Reproducibility 

#=======================================================================
#  Case study
#================================================
# 
# What you should submit
# The goal of your project is to predict the manner in which they 
#  did the exercise. This is the "classe" variable in the training set. 
#  You may use any of the other variables to predict with. 
#   You should create a report describing how you built your model, 
#   how you used cross validation, what you think the expected out of 
#   sample error is, and why you made the choices you did. 
#    You will also use your prediction model to predict 20 different test cases. 
# 
# Peer Review Portion
# Your submission for the Peer Review portion should consist of a link to a 
#  Github repo with your R markdown and compiled HTML file 
#   describing your analysis. Please constrain the text of the 
#   writeup to < 2000 words and the number of figures to be less than 5. 
#    It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
# 
# See the R markdown file for the deployed GitHub Page
#     - JohnsHopkinsMachineLearning-CaseStudy.Rmd 
#

# Open the data files
# 
DATASETPATH <- "./"
list.files(path = DATASETPATH)

myFilename <- paste( DATASETPATH, "pml-training.csv", sep="" )
pml_training <- read.csv(myFilename, stringsAsFactors=FALSE)
nrow( pml_training )

myFilename <- paste( DATASETPATH, "pml-testing.csv", sep="" )
pml_testing <- read.csv(myFilename, stringsAsFactors=FALSE)
nrow( pml_testing )

#  Clean up these Factors at the base level here
#
pml_training$classe  <- as.factor(pml_training$classe)

pml_training$new_window  <- as.factor(pml_training$new_window)
pml_training$user_name  <- as.factor(pml_training$user_name)
pml_testing$new_window  <- as.factor(pml_testing$new_window)
pml_testing$user_name  <- as.factor(pml_testing$user_name)

# Make the cvtd_timestamp into a proper datetime
#

pml_training$cvtd_timestamp <- parse_date_time( pml_training$cvtd_timestamp,
                 c("mdy HM", "dmY HM", "%m/%d/%Y %H:%M:%S %p") )
pml_testing$cvtd_timestamp <- parse_date_time( pml_testing$cvtd_timestamp,
                  c("mdy HM", "dmY HM", "%m/%d/%Y %H:%M:%S %p") )
str(pml_training$cvtd_timestamp )


#========================================================================
#    Cross-Validation
#
#     Create the 'training' and 'testing' datasets from the 
#       'pml-training.csv' dataset.
#     This allows to perform simulated 'out of sample' measurements
#      to measure the progress of our model.
#
#   Random subsample of the pml-training.csv data
#     Two thirds in training and one third in testing.
#
#
inTrain = createDataPartition(pml_training$classe, p =0.67, list=FALSE )
training = pml_training[inTrain,]
testing = pml_training[-inTrain,]
nrow( training ) ; nrow( testing )

#========================================================================
#  Build Regularized Regression model (LASSO) to 
#     find important predictor variables
#
# 
# # Makes sure that 'classe' is a factor
# training$classe  <- as.factor(training$classe)
# testing$classe  <- as.factor(testing$classe)
# summary( training$classe )
# 
# # new_window and user_name  should also be Factors
# training$new_window  <- as.factor(training$new_window)
# training$user_name  <- as.factor(training$user_name)
# 
# testing$new_window  <- as.factor(testing$new_window)
# testing$user_name  <- as.factor(testing$user_name)
# # 


#===========================================================================
# These columns contain a Division by Zero error in the original dataset.
#   Let's remove them for the model building exercises.
#
omitNA_columns <- c(-130,-127, -92, -89, -74, -73, -71, -70, -17, -16, -14, -13 )
# 

cleanTrainingDF <- training[, omitNA_columns]
cleanTestingDF <- testing[, omitNA_columns]
# 
#  str(cleanTrainingDF)
# str(cleanTrainingDF$classe )
# str(cleanTestingDF$classe )
# 
# str(cleanTrainingDF$new_window )
# str(cleanTestingDF$new_window )
# 
# str(cleanTrainingDF$user_name )
# str(cleanTestingDF$user_name )


# # Testing
# date()
# myNames <- names( training)
# training[, 130]
# myNames[130]

#  Use a LASSO - Regularized Regression model
#    use preProcesss() to center and scale the data
#
#  preProcess=c("center", "scale"),
#
myLASSO <-  train( classe ~ ., 
                   data=cleanTrainingDF, 
                   method="glmnet",
                   preProcess=c("center", "scale"),
                   na.action = na.omit,
                   contrasts=contr.treatment )
print( myLASSO )
# 
# myLASSO$bestTune
# myLASSO$bestTune$lambda
# 

# Prediction   -                               not working  TBD
#  Error in model.frame.default(Terms, newdata, 
#  na.action = na.action, xlev = object$xlevels) : 
# 
# pred1 <-  predict( myLASSO, newdata=cleanTestingDF )
# 
# cleanTestingDF$predictedRight <- pred1 == cleanTestingDF$classe
# # view the Cross table and also graph the ACCURACY 
# table( pred1, testing$diagnosis)
# #  Accuracy percentage 
# nrow( cleanTestingDF[ which( cleanTestingDF$predictedRight ==TRUE ), ] ) /
#   nrow( cleanTestingDF )
# #   [1] 0.8414634


#==============================================================
#   Performance measurement
#

# Coefficients  - move toward zero 
myCoef <-  coef( myLASSO$finalModel, myLASSO$bestTune$lambda )


# Use varImp to access the Cofficients and show us the best predictors
#
myVarList  <-  varImp( myLASSO )
myDF <- as.data.frame( myVarList$importance )
myDF$varName <- rownames( myDF )

myDF$OverallImp <-  myDF$A + myDF$B + myDF$C + myDF$D + myDF$E 

# Get the best Predictors and remove timestamps
bestPredictorsDF <- myDF %>%
  filter( OverallImp > 7 & OverallImp < 100 ) %>%
  arrange(OverallImp) 

# filter( ! str_detect( varName, '^cvtd_timestamp' ))  %>%

# Create the composite Formula string 
tmpStr <- paste(bestPredictorsDF$varName, collapse=" + " )
formula_str <- paste( "classe ~ ", tmpStr)
formula( formula_str )
# 
#  stddev_roll_forearm + var_roll_belt + num_window 
#   + var_accel_dumbbell + avg_roll_dumbbell + min_roll_forearm


#==============================================================
# Clear up the NA in our predictor variables
#
#   There are substantial NA counts - set them to the median for now
cleanTrainingDF$stddev_roll_forearm[ is.na(cleanTrainingDF$stddev_roll_forearm)] <-
  median(cleanTrainingDF$stddev_roll_forearm,na.rm=TRUE)
cleanTestingDF$stddev_roll_forearm[ is.na(cleanTestingDF$stddev_roll_forearm)] <-
  median(cleanTestingDF$stddev_roll_forearm,na.rm=TRUE)
pml_testing$stddev_roll_forearm[ is.na(pml_testing$stddev_roll_forearm)] <-
  median(cleanTrainingDF$stddev_roll_forearm,na.rm=TRUE)


cleanTrainingDF$var_roll_belt[ is.na(cleanTrainingDF$var_roll_belt)] <-
  median(cleanTrainingDF$var_roll_belt,na.rm=TRUE)
cleanTestingDF$var_roll_belt[ is.na(cleanTestingDF$var_roll_belt)] <-
  median(cleanTestingDF$var_roll_belt,na.rm=TRUE)
pml_testing$var_roll_belt[ is.na(pml_testing$var_roll_belt)] <-
  median(cleanTrainingDF$var_roll_belt,na.rm=TRUE)

cleanTrainingDF$var_accel_dumbbell[ is.na(cleanTrainingDF$var_accel_dumbbell)] <-
  median(cleanTrainingDF$var_accel_dumbbell,na.rm=TRUE)
cleanTestingDF$var_accel_dumbbell[ is.na(cleanTestingDF$var_accel_dumbbell)] <-
  median(cleanTestingDF$var_accel_dumbbell,na.rm=TRUE)
pml_testing$var_accel_dumbbell[ is.na(pml_testing$var_accel_dumbbell)] <-
  median(cleanTrainingDF$var_accel_dumbbell,na.rm=TRUE)

cleanTrainingDF$avg_roll_dumbbell[ is.na(cleanTrainingDF$avg_roll_dumbbell)] <-
  median(cleanTrainingDF$avg_roll_dumbbell,na.rm=TRUE)
cleanTestingDF$avg_roll_dumbbell[ is.na(cleanTestingDF$avg_roll_dumbbell)] <-
  median(cleanTestingDF$avg_roll_dumbbell,na.rm=TRUE)
pml_testing$avg_roll_dumbbell[ is.na(pml_testing$avg_roll_dumbbell)] <-
  median(cleanTrainingDF$avg_roll_dumbbell,na.rm=TRUE)

cleanTrainingDF$min_roll_forearm[ is.na(cleanTrainingDF$min_roll_forearm)] <-
  median(cleanTrainingDF$min_roll_forearm,na.rm=TRUE)
cleanTestingDF$min_roll_forearm[ is.na(cleanTestingDF$min_roll_forearm)] <-
  median(cleanTestingDF$min_roll_forearm,na.rm=TRUE)
pml_testing$min_roll_forearm[ is.na(pml_testing$min_roll_forearm)] <-
  median(cleanTrainingDF$min_roll_forearm,na.rm=TRUE)

cleanTrainingDF$stddev_pitch_belt [ is.na(cleanTrainingDF$stddev_pitch_belt )] <-
  median(cleanTrainingDF$stddev_pitch_belt ,na.rm=TRUE)
cleanTestingDF$stddev_pitch_belt [ is.na(cleanTestingDF$stddev_pitch_belt )] <-
  median(cleanTestingDF$stddev_pitch_belt ,na.rm=TRUE)
pml_testing$stddev_pitch_belt [ is.na(pml_testing$stddev_pitch_belt )] <-
  median(cleanTrainingDF$stddev_pitch_belt ,na.rm=TRUE)

nrow( cleanTrainingDF )
nrow( cleanTestingDF )

# Caret featurePlot
featurePlot(x=cleanTrainingDF[, c("stddev_pitch_belt", "var_roll_belt", "var_accel_dumbbell",
                                  "avg_roll_dumbbell", "min_roll_forearm")], 
            y=cleanTrainingDF$classe, plot="ellipse" )



#===================================================================
#  Random Forest
#     library(rf)  

# remove num_window variable from our Model Formula - It appears to be causing
#    the model to be overfitted with exact prediction.   This variable may be
#    an administrative field and would not do well in future out-of-sample tests.

# #  Fit a Random forest 
date()
myRF <-  train( classe ~  stddev_roll_forearm +
                  var_roll_belt + var_accel_dumbbell +
                  avg_roll_dumbbell + min_roll_forearm,
                  data=cleanTrainingDF, method="rf" )

#                prox=TRUE )
date()
#   rate is -->    0.293"

# myRF <-  train( formula( formula_str ),
#                 data=cleanTrainingDF, method="rf" )  

#===============================================
# Prediction & measurement
#
pred2 <- predict( myRF, newdata=cleanTestingDF )

nrow( cleanTestingDF )
length( pred2)

# Mark our correct predictions
cleanTestingDF$predictedRight2 <- pred2 == cleanTestingDF$classe
#  testRpart <- predict(fitRpart, newdata = na.omit(dtest))

#  Accuracy percentage 
accuracyRate <- nrow( cleanTestingDF[ which( cleanTestingDF$predictedRight2 ==TRUE ), ] ) /
  nrow( cleanTestingDF )
#   [1] 0.8414634

print( paste( "Our estimated Out-Of-Sample Accuracy rate for RANDOM FOREST is -->   ",
              round( accuracyRate, 4)  ) )


# view the Contingency table and also graph the ACCURACY 
print( "Our Out-Of-Sample Accuracy Contingency Table for RANDOM FOREST" )
table( pred2, cleanTestingDF$classe)

# crosstable( x=cleanTestingDF$classe, y=pred2,  chisq = FALSE )
confusionMatrix( pred2, cleanTestingDF$classe)


#===================================================================
#  Boosting
# library(gbm)    # Boosting
# 
#  Let's try the General Boosting Model 
#
date()
myGBM <-  train( classe ~  stddev_roll_forearm +
                   var_roll_belt + var_accel_dumbbell +
                   avg_roll_dumbbell + min_roll_forearm,
                 data=cleanTrainingDF, method="gbm", verbose=FALSE )
date()

# Prediction 
pred3 <- predict( myGBM, newdata=cleanTestingDF )

nrow( cleanTestingDF )
length( pred3 )

# Mark our correct predictions
cleanTestingDF$predictedRight3 <- pred3 == cleanTestingDF$classe

#  Accuracy percentage 
accuracyRate <- nrow( cleanTestingDF[ which( cleanTestingDF$predictedRight3 ==TRUE ), ] ) /
  nrow( cleanTestingDF )

print( paste( "Our estimated Out-Of-Sample Accuracy rate for GBM BOOSTING is -->   ",
              round( accuracyRate, 4)  ) )


#===================================================================
#  Random sample
# 
#  Let's get a baseline with just randomly generated predictions.
#

myFactorVars <- c( "A", "B", "C", "D", "E" )
sampleSize <- nrow(  cleanTestingDF )

predRandom <- sample( myFactorVars, size=sampleSize, replace = TRUE )

nrow( cleanTestingDF )
length( predRandom )

# Mark our correct predictions
cleanTestingDF$predictedRightRandom<- predRandom == cleanTestingDF$classe

#  Accuracy percentage 
accuracyRate <- nrow( cleanTestingDF[ which( cleanTestingDF$predictedRightRandom ==TRUE ), ] ) /
  nrow( cleanTestingDF )

print( paste( "Our estimated Out-Of-Sample Accuracy rate is for a RANDOM SAMPLE is -->   ",
              round( accuracyRate, 4)  ) )


#=========================================================================
#  Stacking ensemble 
#

#=========================================  Stacking ================
#  Now determine the Stacked accuracy....


qplot( pred2, pred3, colour=classe, data=cleanTestingDF )

# Combine the predictions
#

myFactorVars <- c( "A", "B", "C", "D", "E","F", "G", "H", "I", "J", "K", "M" )
bigClass <- as.factor( myFactorVars )
str(bigClass )
# predictionsDF <- data.frame( pred2, pred3, classe=cleanTestingDF$classe )


# Add more Levels for GAM  training...
tmpClassDF <- cleanTestingDF

str(tmpClassDF$classe  )
levels( tmpClassDF$classe )

levels( tmpClassDF$classe ) <- c( "A", "B", "C", "D", "E","F", "G", "H", "I", "J", "K", "M" )

predictionsDF <- data.frame( pred2, pred3, classe=tmpClassDF$classe )
# 
#   most basis expansion method will fail 
#
# Note: Which terms enter the model in a nonlinear manner is determined by the number of unique
# values for the predictor. For example, if a predictor only has four unique values, most basis expansion
# method will fail because there are not enough granularity in the data. By default, a predictor
# must have at least 10 unique values to be used in a nonlinear basis expansion.

combModFit <- train( classe ~  . ,
                     method="gam",
                     data=predictionsDF )


combPred <- predict( combModFit, predictionsDF)

# Compare Root Sum of squared errors 
# only on quantitative prediction
#  sqrt( sum( (pred1-testing$diagnosis)^2 ) )

# Predict on Validation dataset

pred2V <- predict( myRF, cleanTestingDF)
pred3V <- predict( myGBM, cleanTestingDF)


predVDF <- data.frame( pred2=pred2V, pred3=pred3V )

combPredV <-  predict( combModFit, predVDF  )

validation$predictedRightC <- combPredV == validation$diagnosis
# view the Cross table and also graph the ACCURACY 
table( combPredV, validation$diagnosis)

#  Accuracy percentage 
nrow( validation[ which( validation$predictedRightC ==TRUE ), ] ) /
  nrow( validation )
#   0.8383838  

#   Stacked Accuracy: 0.88 is better than all three other methods








#=========================================================================
#  Predict on pml-testing dataset - 20 predictions of classes
#
#        Use the best model for Prediction 

myPredictions <- predict( myRF, newdata=pml_testing )
myPredictions






#=========================================================================
#  GLM  Linear Regression
#

#  stddev_roll_forearm + var_roll_belt + num_window 
#   + var_accel_dumbbell + avg_roll_dumbbell + min_roll_forearm

myForm=formula( formula_str )

# GLM can only be used on two class outc
myGLM <-  train( classe ~ stddev_roll_forearm + var_roll_belt + num_window,
                data=cleanTrainingDF,
                method="glm" ) 

#                 prox=TRUE )    family = binomial


#  modelFit  <- train( diagnosis.inTrain. ~., data=newData, method="glm" )

# myRF <-  train( classe ~ ., data=cleanTrainingDF, method="rf",
#                 prox=TRUE,
#                 preProcess=c("center", "scale"), na.action = na.omit )
# 
# 
# myRF <- train( classe ~ ., data=cleanTrainingDF, method="rf", prox=TRUE  )


varImp( myRF )

# most important
ggplot( varImp( myRF ))


# get a single tree from the RF 
getTree( myRF$finalModel, k=2)
# Predict on the Random Forest
pred1 <- predict( myRF, vowel.test )

vowel.test$predictedRight <- pred1 == vowel.test$y

# view the Cross table and also graph the ACCURACY 
table( pred1, vowel.test$y )
qplot(pred1, y, colour=predictedRight, data=vowel.test )

#  Check the predictions against the actual values in _test
#
CrossTable( x=wbcd_test_labels, y=pred2,  chisq = FALSE )


table( vowel.test$predictedRight )
# 
# FALSE  TRUE 
# 184   278 

#  Accuracy percentage 
nrow( vowel.test[ which( vowel.test$predictedRight ==TRUE ), ] ) /
  nrow( vowel.test )
#  Accuracy -  
#    [1]  0.6017316 

