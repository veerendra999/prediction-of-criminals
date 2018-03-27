rm(list=ls(all=TRUE))
getwd()
setwd("C:/Users/Hanu/Documents/insofe/hackerearth")

crime <- read.csv("criminal_train.csv")
crime_test = read.csv("criminal_test.csv")

full <- rbind(crime, crime_test)

library(DMwR)
library(tidyr)
library(ggplot2)
library(e1071)
library(tidyverse)
library(dplyr)

library(caret)

str(crime)

#--------------------------------------------train-----------------------------------
crime$Criminal = as.factor(crime$Criminal)

str(crime)

summary(crime)

sum(is.na(crime))

crime2<- arrange(crime, IFATHER)

head(crime2)

set.seed(123)

train_rows<- createDataPartition(crime2$Criminal, p = 0.7, list = F)

pre_train_data<- crime2[train_rows,]

pre_test_data<- crime2[-train_rows, ]

std_m = preProcess(crime2[, !(names(crime2) %in% "Criminal")], method = c("center","scale"))

train_data <- predict(std_m, pre_train_data)
test_data<- predict(std_m, pre_test_data)


summary(crime2)

head(train_data)

#------------------------------------------test------------------------------------
head(crime_test)

str(crime_test)

dim(crime_test)

crime_test<- arrange(crime_test, IFATHER)


crime_test_data <- predict(std_m, crime_test)


summary(crime_test_data)

#-----------------------------------------------------model--------------------------
library(randomForest)

model = randomForest(Criminal~., data = train_data, keep.forest = TRUE, ntree = 50, max_features = "auto", min_samples_leaf = 50)

print(model)


rf_Imp_Attr = data.frame(model$importance)
rf_Imp_Attr = data.frame(row.names(rf_Imp_Attr),rf_Imp_Attr[,1])
colnames(rf_Imp_Attr) = c('Attributes', 'Importance')
rf_Imp_Attr = rf_Imp_Attr[order(rf_Imp_Attr$Importance, decreasing = TRUE),]

print(rf_Imp_Attr)


varImpPlot(model)


#------------------------------------train---------------------------------------------------------

pred_Train = predict(model, 
                     train_data[,setdiff(names(train_data), "Criminal")],
                     type="response", 
                     norm.votes=TRUE)

cm_Train = table("actual"= train_data$Criminal, "predicted" = pred_Train);
accu_Train= sum(diag(cm_Train))/sum(cm_Train)

print(cm_Train)
print(accu_Train)

#---------------------------------train_test--------------------------------------------------------

pred_Test = predict(model, test_data[,setdiff(names(test_data), "Criminal")],
                    type="response", norm.votes=TRUE)
cm_Test = table("actual" = test_data$Criminal, 
                "predicted" = pred_Test);

accu_Test_Imp = sum(diag(cm_Test))/sum(cm_Test)
print(cm_Test)
print(accu_Test_Imp)


confusionMatrix(pred_Test, test_data$Criminal,mode = "prec_recall"  )







#------------------------------------------model---------------------------------------------------

crime$classlabel <- ifelse(crime$Criminal == 0, "1", "0")












#-----------------------------------------test-------------------------------------------------------
pred_Test_crime = predict(model, crime_test_data)


prediction<- write.csv(pred_Test_crime, file = "crime_prediction.csv")



pred_Test_crime

cm_Test = table("actual" = train_data$Criminal, 
                "predicted" = pred_Test_crime);
accu_Test_Imp = sum(diag(cm_Test))/sum(cm_Test)

print(cm_Test)

print(accu_Test_Imp)

confusionMatrix(pred_Test_crime , crime_test_data$Criminal, mode = "prec_recall"  )




