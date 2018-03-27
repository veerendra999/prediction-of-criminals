rm(list=ls(all=TRUE))

library(DMwR)
library(tidyr)
library(ggplot2)
library(e1071)
library(tidyverse)
library(dplyr)
library(caret)


getwd()
setwd("C:/Users/Hanu/Documents/insofe/hackerearth")

crime <- read.csv("criminal_train.csv")
crime_test = read.csv("criminal_test.csv")

crime$classlabel <- ifelse(crime$Criminal == 0, "1", "0")

ts_labels<-crime$classlabel  

str(crime)

summary(crime)

sum(is.na(crime))

crime2<- arrange(crime, IFATHER)

head(crime2)

set.seed(123)

train_rows<- createDataPartition(crime2$Criminal, p = 0.7, list = F)

pre_train_data<- crime2[train_rows,]

pre_test_data<- crime2[-train_rows, ]


std_m = preProcess(crime2[, !(names(crime2) %in% ("Criminal"))], method = c("center","scale"))

train_data <- predict(std_m, pre_train_data)
test_data<- predict(std_m, pre_test_data)
crime_test_data<- predict(std_m, crime_test)


library(xgboost)

train_matrix <- xgb.DMatrix(data = as.matrix(train_data[, !(names(train_data) %in% c("Criminal", "classlabel"))]), 
                            label = as.matrix(train_data[, names(train_data) %in% "Criminal"]))

test_matrix <- xgb.DMatrix(data = as.matrix(test_data[, !(names(test_data) %in% c("Criminal", "classlabel"))]), 
                           label = as.matrix(test_data[, names(test_data) %in% "Criminal"]))

#-------------------------------------------------model---------------------------------------------

modelLookup("xgbTree")

xgb_model_basic <- xgboost(data = train_matrix, max.depth = 2, eta = 1, nthread = 2, nround = 500, objective = "binary:logistic", verbose = 1, early_stopping_rounds = 10)

basic_preds <- predict(xgb_model_basic, test_matrix)

basic_preds_labels <- ifelse(basic_preds < 0.5, 0, 1)

confusionMatrix(basic_preds_labels, test_data$Criminal)

#------------------------------------------------plot------------------------------------------------
variable_importance_matrix <- xgb.importance(feature_names = colnames(train_matrix), model = xgb_model_basic)

xgb.plot.importance(variable_importance_matrix)

#------------------------------------------------test------------------------------------------------

xgb_test<- predict(xgb_model_basic, data.matrix(crime_test_data))

xgb_test


prediction<- write.csv(xgb_test, file = "crime_prediction_xgb.csv")



