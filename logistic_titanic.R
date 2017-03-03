# Laura Niss 3/3/17
# Kaggle Titanic using penalized logistic regression, EM/IRLS for MLE
library(optimbase)
library(matrixcalc)

H_inv <- function(w, rho, lambda, x){
  # inverse Hessian of likelihood function, if Hessian
  # singular, use scalar step size rho
  H.tmp <- -t(x) %*% w %*% x
  if(!is.singular.matrix(H.tmp)){
    H.tmp <- solve(H.tmp-2*lambda*diag(ncol(x)))} 
  else{
    H.tmp <- rho}
  return(H.tmp)
}

IRLS_fit <- function(df,y,rho, lambda){
  # estimate theta using EM algorithm, IRLS
  # essentially gradient ascent
  x <- cbind(1, as.matrix(df))
  theta <- zeros(nx = ncol(x), ny = 1) # initialize theta
  norm <- 100
  i <- 1
  while (norm > 10^(-6) && i < 1000){
    eta <- t(theta) %*% t(x)
    mu <- 1 / (1+ exp(-eta))
    l <- t(x) %*% t(y - mu) - 2*lambda*theta # penalised likelihood
    w <- diag(as.vector(mu*(1-mu)))
    h_inv <- H_inv(w,rho, lambda, x) # inverse Hessian
    if (is.null(dim(h_inv))){
      theta_update <- theta + h_inv * l }
    
    else {
      theta_update <- theta - h_inv %*% l} 
    
    norm <- norm(theta-theta_update, type='F')
    theta <- theta_update
    i = i + 1
  }
  return(theta)
}

Logistic <- function(x, theta){
  # get prediction from test set
  x <- cbind(1, x)
  test <- rep(0,nrow(x))
  for(i in 1:nrow(x)){
    if(1/(1+exp(-t(theta)%*%t(x[i,]))) >= .5){
      test[i] = 1
    }
    else{
      test[i] = 0
    }
  }
  return(test)
}

error <- function(test, true){
  # check error rate
  df <- data.frame(true,test)
  error <- 1 - length(subset(df$test, df$true == df$test))/nrow(df)
  return(error)
}

# set working directory
setwd("~Dropbox/R/Titanic")
#import train.csv to train
train <- read.csv("~/Dropbox/R/Titanic/train.csv", stringsAsFactors=FALSE)
View(train)
#import test.csv into test  
test <- read.csv("~/Dropbox/R/Titanic/test.csv", stringsAsFactors=FALSE)
View(test)

# Some minimal data cleaning
embarked <- function(df){
  df$Embarked[df$Embarked == 'S'] = 1
  df$Embarked[df$Embarked == 'C'] = 2
  df$Embarked[df$Embarked == 'Q'] = 3
  df$Embarked[df$Embarked == ''] = 0
  return(df$Embarked)
}
# Fill missing continuous with median
median = median(train$Age, na.rm = TRUE)
train$Age[is.na(train$Age)] <- median
median = median(train$Fare, na.rm = TRUE)
train$Fare[is.na(train$Fare)] <- median

median = median(test$Age, na.rm = TRUE)
test$Age[is.na(test$Age)] <- median
median = median(test$Fare, na.rm = TRUE)
test$Fare[is.na(test$Fare)] <- median

# Make categorical numeric
train$Embarked = embarked(train)
test$Embarked = embarked(test)

# Make categorical binary
train$Cabin[train$Cabin == ''] = 0
train$Cabin[train$Cabin != 0] = 1
train$Sex[train$Sex == 'female'] = 1
train$Sex[train$Sex != 1] = 0

test$Cabin[test$Cabin == ''] = 0
test$Cabin[test$Cabin != 0] = 1
test$Sex[test$Sex == 'female'] = 1
test$Sex[test$Sex != 1] = 0

# Variables kept for fit
train_df = train[c("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", 'Cabin', "Embarked")]
test_df = test[c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",'Cabin', "Embarked")]

# make everything numeric so can use matrix
train_df$Sex = as.numeric(unlist(train_df$Sex))
train_df$Embarked = as.numeric(unlist(train_df$Embarked))
train_df$Cabin = as.numeric(unlist(train_df$Cabin))

test_df$Sex = as.numeric(unlist(test_df$Sex))
test_df$Embarked = as.numeric(unlist(test_df$Embarked))
test_df$Cabin = as.numeric(unlist(test_df$Cabin))

# Functions to tranform
tranform <- function(df){
  # logistic transformation
  tmp <- log(df[,-ncol(df)]+1)
  return(data.frame(tmp, df[,ncol(df)]))
}

standardize <- function(df){
  # standardize data
  return(data.frame(scale(df[,-ncol(df)]), df[,ncol(df)]))
}

# Split training data into test and train to check error rates
smp_size <- floor(0.75 * nrow(train_df))

# set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(train_df)), size = smp_size)

train_tmp <- train_df[train_ind, ]
test_tmp <- train_df[-train_ind, ]

x <- transform(train_tmp[,-1]) # training data without class
y <- train_tmp[,1] # class of training data

# fit logistic regression, change rho, lambda to test fits
theta = IRLS_fit(x, y, 1, 10) 

x <- transform(test_tmp[,-1]) # testing data without class
prediction = Logistic(x, theta) # get prediction of testing data

err = error(prediction, test_tmp[,1]) # error rate 
err

# submission
submit <- data.frame(PassengerId = test$PassengerId, Survived = prediction)
write.csv(submit, file = "logistic.csv", row.names = FALSE)
