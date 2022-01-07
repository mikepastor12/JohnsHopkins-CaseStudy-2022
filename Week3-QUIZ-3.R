#=======================================================================================
#  Week3-QUIZ-Main.R
#
#   Predicting with Trees
#
#     Mike Pastor  January 5, 2022

rm( list = ls() )    # Clear Environment objects


#====================================================================
#  Load the Packages
#
#  install.packages( "forecast" )
library(AppliedPredictiveModeling)
library(caret)

library(ElemStatLearn)
library(rpart)
library(randomForest)
library(gbm)    # Boosting

library( quantmod )   # forecasting data
library( forecast )   # forecasting data


library(MASS)
library(miniUI)
library(klaR)  # Naive Bayes

library(dplyr)
library(ggplot2)

library(RANN)

library(Hmisc)
library(gridExtra)
library(splines)

# Bagging code
library(party)

library(ISLR)


#   QUIZ 3  examples
library(AppliedPredictiveModeling)

data(segmentationOriginal)
library(caret)


# 
# 1. Subset the data to a training set and testing set 
#     based on the Case variable in the data set. 

# 2. Set the seed to 125 and fit a CART model with the rpart method 
#    using all predictor variables and default caret settings. 

# 3. In the final model what would be the final model prediction for cases with the following variable values:
#   a. TotalIntench2 = 23,000; FiberWidthCh1 = 10; PerimStatusCh1=2 
# b. TotalIntench2 = 50,000; FiberWidthCh1 = 10;VarIntenCh4 = 100 
# c. TotalIntench2 = 57,000; FiberWidthCh1 = 8;VarIntenCh4 = 100 
# d. FiberWidthCh1 = 8;VarIntenCh4 = 100; PerimStatusCh1=2 
# 
# 
# a. PS 
# b. WS 
# c. PS
# d. WS 
# 
set.seed( 125 )

soIndex = createDataPartition(segmentationOriginal$Class, p = 0.70,list=FALSE)
training = segmentationOriginal[soIndex,]
testing = segmentationOriginal[-soIndex,]

label( training )
# 
# myModel <- rpart( Class ~ TotalIntenCh2 + FiberWidthCh1 + PerimStatusCh1,
#                   data = training )
# # 
# myModel <-  train( Class ~ .,
#                    method="rpart",  data = training )
# print( myModel$finalModel )
# 
# plot( myModel$finalModel, uniform=TRUE, main="mytest")
# text( myModel$finalModel, use.n=TRUE, all=TRUE, cex=.8 )

# 
myModel <-  train( Class ~ TotalIntenCh2 + FiberWidthCh1 + PerimStatusCh1 + VarIntenCh4,
                   method="rpart",  data = training )

# pred1 <-  predict( myModel,  newdata = testing)

# setup prediction and execute
myParms <- data.frame( TotalIntenCh2 =23000, FiberWidthCh1 =10,
                       PerimStatusCh1 =2, VarIntenCh4=0)
predict( myModel, newdata=myParms )

# setup prediction and execute
myParms <- data.frame( TotalIntenCh2 =50000, FiberWidthCh1 =10,
                       PerimStatusCh1 =0, VarIntenCh4=100)
predict( myModel, newdata=myParms )

# setup prediction and execute
myParms <- data.frame( TotalIntenCh2 =57000, FiberWidthCh1 =8,
                       PerimStatusCh1 =0, VarIntenCh4=100)
predict( myModel, newdata=myParms )

# setup prediction and execute
myParms <- data.frame( FiberWidthCh1 =8,
                       PerimStatusCh1 =2, VarIntenCh4=100, TotalIntenCh2 =0)
predict( myModel, newdata=myParms )

#============================================================================

#  library(pgmm)

load( "olive.rda" )

olive <- olive[,-1]

newData1 <-  as.data.frame( t( colMeans( olive )))
# 
# Area     Palmitic Palmitoleic  Stearic    Oleic Linoleic Linolenic Arachidic Eicosenoic
# 4.59965 1231.741    126.0944 228.8654 7311.748  980.528  31.88811   58.0979   16.28147


#============================================================================

library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]



set.seed(13234)
# 
# Then set the seed to 13234 and fit a 
# logistic regression model  (method="glm", be sure to specify family="binomial") 
# with Coronary Heart Disease (chd) as the outcome and age at onset, 
# current alcohol consumption, obesity levels, cumulative tabacco, t
# ype-A behavior, and low density lipoprotein cholesterol as predictors. 
# 
# Calculate the misclassification rate for your model using this function 
# and a prediction on the "response" scale:
# 

missClass = function(values,prediction){
  sum(((prediction > 0.5)*1) != values)/length(values)}

# trainSA_DF <-  as.data.frame( trainSA )

trainSA$chd <-  as.factor(trainSA$chd )
testSA$chd <-  as.factor(testSA$chd )


myModel <-  glm( chd ~ ., data=trainSA, family="binomial")

# myModel <-  train( chd ~ ., method="glm", family="binomial", data=trainSA )

pred1 <- predict( myModel, testSA )

str( testSA$chd)
str( pred1)

missClass( testSA$chd,   pred1  )

#  missClass( as.numeric( testSA$chd),  as.numeric( pred1 ) )

pred2 <- predict( myModel, trainSA )
missClass( trainSA$chd,  pred2  )


#============================================================================

library(ElemStatLearn)
data(vowel.train)
data(vowel.test)

set.seed(33833)

vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)
str( vowel.train$y )


myRF <-  randomForest( y ~ ., data=vowel.train )

varImp( myRF )


# again

myRF2 <-  train( y ~ ., data=vowel.train, method="rf", prox=TRUE  )

varImp( myRF2 )

# get a single tree from the RF 
getTree( myRF2$finalModel, k=2)

# Predict on the Random Forest
pred1 <- predict( myRF2, vowel.test )

vowel.test$predictedRight <- pred1 == vowel.test$y

# view the Cross table and also graph the ACCURACY 
table( pred1, vowel.test$y )

qplot(pred1, y, colour=predictedRight, data=vowel.test )


#===================================================================
#  Boosting
library(gbm)    # Boosting

myGBM <-  train( y ~ ., data=vowel.train, method="gbm", verbose=FALSE  )

varImp( myGBM )

#=====================================================================
#  Naive Bayes 

library(MASS)
library(miniUI)
library(klaR)  # Naive Bayes

myNB <-  train( y ~ ., data=vowel.train, method="nb", verbose=FALSE  )

varImp( myNB )

# Predict on the Random Forest
pred1 <- predict( myNB, vowel.test )

vowel.test$predictedRight <- pred1 == vowel.test$y

# view the Cross table and also graph the ACCURACY 
table( pred1, vowel.test$y )

qplot(pred1, y, colour=predictedRight, data=vowel.test, main="Accuracy map - mlp" )
# 
# #=============================================================================
# 
# #  Week 4  - Regularized Regression
# 
# Caret -
#   ridge
#   lasso
#   relaxo
#   
#   
  
#===========================================================================
#
#     Forecasting - time series data

library( quantmod )

from.date <- as.Date( "01/01/08",  format="%m/%d/%y")
to.date <- as.Date( "12/31/13",  format="%m/%d/%y")

# Get the data online and store it in "GOOG"
getSymbols("ADP", src="yahoo", from=from.date, to=to.date )

GOOG <- ADP
head( GOOG )

#  summarize and create a time-series object...
#
mGoog <- to.monthly( GOOG )
googOpen <- Op( mGoog )
ts1 <-  ts( googOpen, frequency=12 )
plot( ts1,  xlab="years",   ylab="Stock Price")

# Decompose the time series
#
dec1 <- decompose( ts1)
plot( dec1, xlab="Years+1")

# Build your data partitions in window() 

ts1Train <-  window( ts1, start=1, end=5 )
ts1Test <-  window( ts1, start=5, end=(7-0.01) )

#  Forecast with a moving average
library( forecast )   # forecasting data

plot( ts1Train )
lines(  ma( ts1Train, order=3 ),    col="red")

# exponential smoothing  - forecast()

ets1 <- ets( ts1Train, model="MMM" )
fcast <- forecast( ets1 )

plot( fcast )
lines( ts1Test, col="red")

# accuracy  - RMSE 
accuracy( fcast, ts1Test )

#                   ME     RMSE      MAE        MPE      MAPE      MASE       ACF1 Theil's U
# Training set 0.1508195 1.932842 1.512502  0.2785369  4.037791 0.2727487 0.02337597        NA
# Test set     7.4808535 9.828036 7.744986 12.5139733 13.099393 1.3966494 0.83384300  4.754473



