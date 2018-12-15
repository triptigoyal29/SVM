################### Handwritten Digit Recognition using SVM #################################
#############################################################################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation & EDA
# 4. Model Building using SVM
# 5. Hyperparameter tuning using Cross Validation
# 6. Model Evaluation and Interpretations

###########################################################################################
### 1. Business Understanding
###########################################################################################

# A classic problem in the field of pattern recognition is that of handwritten digit recognition.
# Suppose that you have an image of a digit submitted by a user via a scanner, a tablet, or other 
# digital devices. 
# The goal is to develop a model that can correctly identify the digit (between 0-9) written 
# in an image. 

###########################################################################################
### 2. Data Understanding
###########################################################################################

## Loading necessary libraries
library(caret)   
library(kernlab)  
library(dplyr)
library(readr)
library(ggplot2)
library(gridExtra)
library(DescTools)
library(doParallel)

## Loading datasets into the R environment.
mnist_train <- read.csv("mnist_train.csv", header = FALSE)
mnist_test <- read.csv("mnist_test.csv", header = FALSE)

## Describe and understand datasets

dim(mnist_train)  ## 60000 observations of 785 variables
head(mnist_train)

dim(mnist_test)   ## 10000 observations of 785 variables
head(mnist_test)

## Check for the datatypes of all the attributes in both train and test datasets

str(mnist_train)
mnist_train[ , lapply(mnist_train, is.numeric) == FALSE ]  ## All attributes are of integer datatype

str(mnist_test)
mnist_test[ , lapply(mnist_test, is.numeric) == FALSE ]    ## All attributes are of integer datatype


## Check for NAs in train and test
sapply(mnist_train, function(x) sum(is.na(x)))    ## No Missing Values
sapply(mnist_test, function(x) sum(is.na(x)))    ## No Missing Values

## Check for duplicated rows
sum(duplicated(mnist_train))          ## Return 0, hence no duplicated data
sum(duplicated(mnist_test))           ## Return 0, hence no duplicated data

## Checking for outliers

# According to Problem statement, there are 28x28 pixel images of digits (contributing to 
# 784 columns) as well as one extra label column. Since, each pixel provides the density of
# black in that pixel, therefore value ranges from 0 to 255.
# Hence, checking the min and max values in the dataset for any outliers.
max(mnist_train)    ## Returns 255
min(mnist_train)    ## Returns 0

max(mnist_test)     ## Returns 255
min(mnist_test)     ## Returns 0

###########################################################################################
### 3. Data Preparation and EDA
##########################################################################################

## Target Variable

# Lets change the column name of the first column, as it signifies the Class
colnames(mnist_train)[1] <- "digit"
colnames(mnist_test)[1] <- "digit"

# Check for the range of digit
summary(factor(mnist_train$digit))   ## All values in the range of (0-9)
summary(factor(mnist_test$digit))    ## All values in the range of (0-9)

# Plotting digit target variable to understand the distribution in train data set
ggplot(mnist_train, aes(x = as.factor(mnist_train$digit), fill = factor(digit))) + 
             geom_bar() + geom_text(stat = 'count', aes(label=..count..), vjust=-0.3) +
             xlab("Digit labels") + ylab("Count of Digit Labels")

Desc(mnist_train$digit)

# Plotting digit target variable to understand the distribution in test data set
ggplot(mnist_test, aes(x = as.factor(mnist_test$digit), fill = factor(digit))) + 
  geom_bar() + geom_text(stat = 'count', aes(label=..count..), vjust=-0.3) +
  xlab("Digit labels") + ylab("Count of Digit Labels")

Desc(mnist_test$digit)

# Converting digit target variable to factor variable
mnist_train$digit <- as.factor(mnist_train$digit)
mnist_test$digit <- as.factor(mnist_test$digit)

## NOTE: I have not taken care of Scaling as all the attributes are in the same range of [0-255].
## Also, if we use Scale function, we get NaN for many values. This is because all the columns
## with "Zero" in them have StdDev = 0 and the Scale() function divides by Standard Deviation 
## for calcualtion of scale.


## Since this is Pattern Recognition dataset, all the columns are important.
## But, Train and test datasets still have the problem of relatively large number of features(columns).
## After exploring many methods of Dimension Reduction online on analytics websites and forums,
## I realized that PCA (Principal Component Analysis) can be used to effectively address this problem.
## PCA is a linear transformation algorithm that seeks to project our original features of our
## data onto a smaller set of features, while retaining most of the information.
## But, not including PCA here as it might be out of the scope of this assignment.

###########################################################################################
### 4. Model Building using SVM
##########################################################################################
## Since the dataset is very large, we will sample the data to get 15% of train and test data.

# Set the seed to ensure repeatability of results
set.seed(1000)

train.indices <- sample(1:nrow(mnist_train), 0.15*nrow(mnist_train))
train <- mnist_train[ train.indices, ]

test.indices <- sample(1:nrow(mnist_test), 0.15*nrow(mnist_test))
test <- mnist_test[test.indices, ]

dim(train)     ## 9000 Observations of 785 Variables
dim(test)      ## 1500 Observations of 785 Variables


## 4.1 Building Linear Model with Default Parameters

Model_linear <- ksvm(digit~ ., data = train, scale = FALSE, kernel = "vanilladot")
# Please note that we get a Warning message here which says 'Cannot scale data'. We can ignore this.
# This warning message is due to scaling issues since pixel data has very low variance and most values 
# nearing zero.
Model_linear
# Number of Support Vectors : 2497 
# Classification parameter : cost C = 1 

Eval_linear <- predict(Model_linear, test)
confusionMatrix(Eval_linear, test$digit)
# Overall Model Accuracy : 0.9127          
# It seems even Sensitivity and Specificity is also reasonably well for almost all the Classes.


## 4.2 Building Polynomial Non-Linear SVM Model with Default Parameters
## Note : Performance of Polynomial is almost same as Linear Model, and also not better
## than RBF

Model_poly <- ksvm(digit~ ., data = train, scale = FALSE, kernel = "polydot")
Model_poly
# Number of Support Vectors : 2497 
# Classification parameter : cost C = 1 

Eval_poly <- predict(Model_poly, test)
confusionMatrix(Eval_poly, test$digit)
# Overall Model Accuracy : 0.9127          
# It seems even Sensitivity and Specificity is also reasonably well for almost all the Classes.


## 4.3 Building RBF Non-Linear SVM Model with Default Parameters
Model_RBF <- ksvm(digit~ ., data = train, scale = FALSE, kernel = "rbfdot")
Model_RBF
# Number of Support Vectors : 3514 
# Classification parameter : cost C = 1 
# Hyperparameter : sigma =  1.64992009652402e-07 
# Training error : 0.018778 

Eval_RBF <- predict(Model_RBF, test)
confusionMatrix(Eval_RBF, test$digit)
# Overall Model Accuracy : 0.9527          
# It seems even Sensitivity and Specificity are also better for almost all the Classes as compared 
# to Linear Model.

############### Default Parameter Model Interpretation ##################################
# On comparing the Accuracy, Sensitivity and Specifity values of all 3 above models, we see
# there is a significant increase in the model performance of Gaussian Radial Basis Kernel Model
# (RBF Model)
# Overall Model Accuracy : 0.9527 for RBF Model which is better than Accuracy : 0.9127 for Linear Model
# Even, there is significant increase in Sensitivity and Specificity for RBF model.
# Hence, we will go ahead with RBF model for Hyperparameter Tuning.
#########################################################################################

###########################################################################################
# 5. Hyperparameter tuning using Cross Validation
##########################################################################################

# We will do Hyperparameter tuning by performing 3-fold Cross Validation on training dataset
# Since our dataset is huge with relatively large number of Dimensions, we have kept the number 
# of values of sigma and C also as 3.
# Computational time is high for Cross Validation as this is a repetitve process and it computes
# for all the possible combinations of sigma and C.

## Including parallel processing to reduce processing time.
cl = makeCluster(detectCores())
registerDoParallel(cl)

trainControl <- trainControl(method = "cv", number = 3, verboseIter = TRUE)
# number - Number of folds
# method - Cross Validation

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy
metric <- "Accuracy"

set.seed(100)

## Since the Default parameters for RBF model above are sigma =  1.64992009652402e-07 and 
## Cost C = 1 and we are getting reasonably better performance of the default RBF model.
## Just varying default Hyperparameters Sigma and C by some amount so that values remain 
## closer to default parameters.

# Cross Validation Folds = 3
# Range of Sigma = (2.636e-7, 3.636e-7, 4.636e-7)
# Range of C = (1, 2, 3)

grid <- expand.grid(.sigma = c(2.636e-7, 3.636e-7, 4.636e-7), .C=c(1,2,3))

SVM_non_linear_RBF_tuned.fit <- train(digit~ ., data = train, method = "svmRadial",
                                      metric = metric, tuneGrid = grid, trControl = trainControl)
# Fitting sigma = 3.64e-07, C = 3 on full training set

SVM_non_linear_RBF_tuned.fit

# Support Vector Machines with Radial Basis Function Kernel 
#
# 9000 samples
# 784 predictor
# 10 classes: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 
#
# No pre-processing
# Resampling: Cross-Validated (3 fold) 
# Summary of sample sizes: 5999, 6002, 5999 
# Resampling results across tuning parameters:
#  
# sigma      C  Accuracy   Kappa    
# 2.636e-07  1  0.9593335  0.9547881
# 2.636e-07  2  0.9633332  0.9592342
# 2.636e-07  3  0.9643335  0.9603466
# 3.636e-07  1  0.9619998  0.9577526
# 3.636e-07  2  0.9644444  0.9604700
# 3.636e-07  3  0.9651111  0.9612110
# 4.636e-07  1  0.9631108  0.9589882
# 4.636e-07  2  0.9646663  0.9607170
# 4.636e-07  3  0.9646662  0.9607168
#
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were sigma = 3.636e-07 and C = 3.

plot(SVM_non_linear_RBF_tuned.fit)
# From the plot, its visible that the highest Model accuracy 0.9651111 is obtained when 
# sigma = 3.636e-07 and C = 3.

#########################################################################################
# Building the Final RBF Model with tuned Hyperparamters Sigma and C
#########################################################################################

FinalModel_RBF <- ksvm(digit~ ., data = train, scale = FALSE, kernel = "rbfdot",
                       C = 3, kpar = list(sigma=3.636e-07))

FinalModel_RBF
# Number of Support Vectors : 4151 
# Classification parameter : cost C = 3 
# Hyperparameter : sigma =   3.636e-07 
# Training error : 0.000444 


# Final Model evaluation on Training data
FinalModelEval_RBF_train <- predict(FinalModel_RBF, train)
confusionMatrix(FinalModelEval_RBF_train, train$digit)
# Overall Model Accuracy on Training data: 0.9996          

# Final Model evaluation on unseen Sample test data (15% Test Data)
FinalModelEval_RBF_test <- predict(FinalModel_RBF, test)
confusionMatrix(FinalModelEval_RBF_test, test$digit)
# Overall Model Accuracy on 15% Sample Test Data : 0.9633          
# It seems even Sensitivity and Specificity are also better for almost all the Classes as compared 
# to default RBF Model.

# Final Model Evaluation on Full test data without Sample
FinalModelEval_RBF_mnist_test <- predict(FinalModel_RBF, mnist_test)
confusionMatrix(FinalModelEval_RBF_mnist_test, mnist_test$digit)
# Overall Model Accuracy on Complete Test Unseen data (without Sample) : 0.966
# Overall Accuracy of our Final Model is slightly better on complete mnist_test data set 
# as compared to 15% Sample data. 


## From above Final Model results on Train and Test data, Model's accuracy has improved over 
## default RBF model. Since, there is not much difference between train and unseen data 
## accuracy results i.e 0.9996 and 0.9633 Accuracy respectively, our Final model is not a 
## candidate of Overfitting. Also, its performing quite well on unseen data

stopCluster(cl)

################################ SUMMARY ################################################
# 
# Following are the findings from modelling exercise:
#
# For Models with Default parameters-
# Overall Accuracy for Linear Model : 0.9127 
# Overall Model Accuracy for Non-Linear Polynomial Model : 0.9127 
# Overall Model Accuracy for Non-Linear RBF Model : 0.9527 
#
# For FINAL MODEL,Non-Linear RBF Model with tuned Hyperparameters - 
# Overall Final Model Accuracy on Training data for : 0.9996
# Overall Final Model Accuracy on 15% Sample data : 0.9633    
# Overall Final Model Accuracy on Complete Test Unseen data (without Sample) : 0.966
#
# Our FINAL Model has maximum Accuracy as compared to models with default parameters.
# The Accuracy values of Final Model are comparable on Train and Test Datasets.
# Hence, Final Model is not Overfitted and is acceptable.
# The Final RBF model obtained has the following tuned Hyper parameters
# sigma = 3.636e-07 
# Misclassification Cost C = 3.
#
# Please note that there is still scope of improvement in deriving a better model.
# As stated above, we have not taken care the Dimension reduction. We can use several Dimension
# reduction tecniques like PCA to have a better model.
# Due to limitation of hardware resources and reducing computation time, we have considered lesser 
# range of Hyperparameters Sigma and C (3 each). We can further work to fine tune the Hyper parameters
# by introducing more range of Sigma and C.
#
###########################################################################################

