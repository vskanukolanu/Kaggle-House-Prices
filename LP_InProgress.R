##########################################In Progress Work##########################################
###################################################################################################
sapply(train,function(x) {sum(is.na(x))})
train_NA<- train[rowSums(is.na(train)) > 0,]
train_Non_NA<-na.exclude(train)
t.test(train_NA$ApplicantIncome,train_Non_NA$ApplicantIncome)

#Testing for MCAR of NAs in Credit History
#Little's MCAR Test
LittleMCAR(train)

#Testing using Logistic Regression
train$BD_CreditHistory<-ifelse(is.na(train$Credit_History),"1","0")
train1<-train[,-10]
TMCAR_CH<-glm(as.factor(train1$BD_CreditHistory)~.,data=train1,family=binomial(link="logit"),na.action=na.exclude)
summary(TMCAR_CH)

#Testing for MCAR of NAs in Loan Amount
#Little's MCAR Test
LittleMCAR(train)

#Testing using Logistic Regression
train$BD_LoanAmount<-ifelse(is.na(train$LoanAmount),"1","0")
train2<-train[,-c(8,13)]
TMCAR_LA<-glm(as.factor(train2$BD_LoanAmount)~.,data=train2,family=binomial(link="logit"),na.action=na.exclude)
summary(TMCAR_LA)

#Testing for MCAR of NAs in Self Employed
#Little's MCAR Test
LittleMCAR(train)

#Testing using Logistic Regression
train$BD_SelfEmployed<-ifelse(is.na(train$Self_Employed),"1","0")
train3<-train[,-c(5,13,14)]
TMCAR_SE<-glm(as.factor(train3$BD_SelfEmployed)~.,data=train3,family=binomial(link="logit"),na.action=na.exclude)
summary(TMCAR_SE)

#Testing for MCAR of NAs in Loan Amount Term
#Little's MCAR Test
LittleMCAR(train)

#Testing using Logistic Regression
train$BD_Term<-ifelse(is.na(train$Loan_Amount_Term),"1","0")
train4<-train[,-c(9,15,13,14)]
TMCAR_Term<-glm(as.factor(train4$BD_Term)~.,data=train4,family=binomial(link="logit"),na.action=na.exclude)
summary(TMCAR_Term)

#Testing for MCAR of NAs in Loan Amount Term
#Little's MCAR Test
LittleMCAR(train)

#Testing using Logistic Regression
train$BD_Dependents<-ifelse(is.na(train$Dependents),"1","0")
train5<-train[,-c(3,15,13,14,16)]
TMCAR_Dependents<-glm(as.factor(train5$BD_Dependents)~.,data=train5,family=binomial(link="logit"),na.action=na.exclude)
summary(TMCAR_Dependents)
md.pattern(train[,1:11])

aggr_plot <- aggr(train[,1:11], col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(train[,1:11]), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
train$Credit_History<-as.factor(train$Credit_History)
train.imp <- missForest(as.data.frame(train[,1:12]), maxiter = 20,ntree = 50)

sapply(train.imp,function(x) {sum(is.na(x))})
sapply(train.imp,summary)
train_imp<-as.data.frame(train.imp$ximp)
#Change 3+ factor value to 3 in Dependents
train_imp$Dep<-revalue(train_imp$Dependents, c("3+"="3"))

#Calculate EMI and EMI to Income ratio for each case
train_imp$emi<-train_imp$LoanAmount*(1+(9.5*train_imp$Loan_Amount_Term)/1200)
train_imp$rc<-train_imp$emi/(train_imp$ApplicantIncome+train_imp$CoapplicantIncome)
attach(train_imp)

#Train a logistic regression model
model_imp1<-glm(Loan_Status~Education+Self_Employed+rc+Credit_History+Property_Area,data=train_imp,family = binomial(link="logit"),maxit=100)
model_imp2<-glm(Loan_Status~.,data=train_imp,family = binomial(link="logit"),maxit=100)
model_imp3<-glm(Loan_Status~rc+Credit_History+Property_Area,data=train_imp,family = binomial(link="logit"),maxit=100)

model_imp4<-glm(Loan_Status~Education*Self_Employed+rc+Credit_History+Property_Area,data=train_imp,family = binomial(link="logit"),maxit=100)

New_test<-as.data.frame(as.matrix(sparse_matrix[,c(13,20,19,10)]))
attach(New_test)
model_imp5<-glm(Loan_Status~Credit_History+rc+CoapplicantIncome,data=New_test,family = binomial(link="logit"),maxit=100)





#Predict training data responses using the above model
pred_imp<-predict.glm(model_imp4,newdata = NULL,type = c("response"))

#Set the classification boundary as 0.5
predval_imp<-ifelse(pred_imp>=0.5,"Y","N")

#Generate a confusion matrix
confusionMatrix(predval_imp,Data$Loan_Status)


#Read test data
test<-read.csv("Test.csv",header = TRUE)
test$Credit_History<-as.factor(test$Credit_History)
test.imp <- missForest(as.data.frame(test[,-1]), maxiter = 20,ntree = 50)

test_imp<-as.data.frame(test.imp$ximp)
#Change 3+ factor value to 3 in Dependents
train_imp$Dep<-revalue(train_imp$Dependents, c("3+"="3"))

#Calculate EMI and EMI to Income ratio for each case
test_imp$emi<-test_imp$LoanAmount*(1+(9.5*test_imp$Loan_Amount_Term)/1200)
test_imp$rc<-test_imp$emi*12/(test_imp$ApplicantIncome+test_imp$CoapplicantIncome)
attach(train_imp)

#Predict on test data using the trained logit model
test_pred<-predict.glm(model_imp4,newdata = test_imp[,c("Education","Self_Employed","rc","Credit_History","Property_Area")],type = c("response"))
test_predval<-ifelse(test_pred>=0.5,"Y","N")

#Write the solution file
Final<-cbind.data.frame(test$Loan_ID,test_predval)
write.csv(Final,file="Solution File_V2.csv")


#############################################################################################################################
##############################XGBoost Algorithm############################################################################

#Clear the memory
rm(list=ls())

#Load libraries
library(car)
library(base)
library(dplyr)
library(sqldf)
library(base)
library(compare)
library(gtools)
library(stringdist)
library(plyr)
library(caret)
library(BaylorEdPsych)
library(mvnmle)
library(mice)
library(VIM)
library(missForest)
require(xgboost)
require(Matrix)
require(data.table)

#Set the directory
setwd("C:/Users/Sravan/Desktop/My Learning Docs") #Place only the data files in this folder

#Read the training data
Data<-read.csv("Data.csv",header = TRUE,na.strings = c("","NA"))

#Exclude NAs
#Data<-na.exclude(Data)

#Remove the Loan_ID
train<-Data[,-1]

#Run summary on all variables
sapply(train,summary)

#Missing value Imputation using missForest
train$Credit_History<-as.factor(train$Credit_History)
train.imp <- missForest(as.data.frame(train[,1:12]), maxiter = 20,ntree = 50)

train_imp<-as.data.table(train.imp$ximp)
sapply(train_imp,function(x) {sum(is.na(x))})
sapply(train_imp,summary)

#Change 3+ factor value to 3 in Dependents
train_imp$Dep<-revalue(train_imp$Dependents, c("3+"="3"))

#Calculate EMI and EMI to Income ratio for each case
train_imp$emi<-train_imp$LoanAmount*(1+(9.5*train_imp$Loan_Amount_Term)/1200)
train_imp$rc<-train_imp$emi*12/(train_imp$ApplicantIncome+train_imp$CoapplicantIncome)
train_imp[,rcgroup := as.factor(ceiling(rc))]
attach(train_imp)
str(train_imp)

sparse_matrix <- sparse.model.matrix(train_imp$Loan_Status~.-1, data = train_imp)
head(sparse_matrix)


output_vector <- train_imp[,Loan_Status] == "Y"

#bst <- xgboost(data = sparse_matrix, label = output_vector, max_depth = 4,
eta = 0.05, nthread = 2, nrounds = 2,objective = "binary:logistic", verbose = 1, eval.metric="auc")

importance <- xgb.importance(feature_names = colnames(sparse_matrix), model = bst)



#bst1 <- xgboost(data = New_test, label = output_vector, max_depth = 4,
eta = 0.01, nthread = 10, nrounds = 10,objective = "binary:logistic", verbose = 1, eval.metric="auc")
#importance1 <- xgb.importance(feature_names = colnames(New_test), model = bst1)

best_param = list()
best_seednumber = 1234
best_auc = 0
best_auc_index = 0

param <- list(objective = "binary:logistic",
              eval_metric = "auc",
              max_depth = 5,
              eta = 0.1,
              gamma = 0, 
              subsample = 0.8,
              colsample_bytree = 0.8, 
              min_child_weight = 1,
              max_delta_step = sample(1:10, 1))



mdcv <- xgb.cv(data=sparse_matrix,label = output_vector, params = param, nthread=4, 
               nfold=10, nrounds=100,
               verbose = T, early.stop.round=50, maximize=FALSE)

max_auc = max(mdcv[, test.auc.mean])
max_auc_index = which.max(mdcv[, test.auc.mean])


best_auc = max_auc
best_auc_index = max_auc_index
best_seednumber = seed.number
best_param = param


nround <- best_auc_index
set.seed(best_seednumber)
md <- xgboost(data=New_test,label = output_vector , params=best_param, nrounds=nround, nthread=6)

#####################################We are overfitting XGB########################################################
#Read test data
test<-read.csv("Test.csv",header = TRUE)
test$Credit_History<-as.factor(test$Credit_History)
test.imp <- missForest(as.data.frame(test[,-1]), maxiter = 20,ntree = 50)

test_imp<-as.data.frame(test.imp$ximp)
#Change 3+ factor value to 3 in Dependents
train_imp$Dep<-revalue(train_imp$Dependents, c("3+"="3"))

#Calculate EMI and EMI to Income ratio for each case
test_imp$emi<-test_imp$LoanAmount*(1+(9.5*test_imp$Loan_Amount_Term)/1200)
test_imp$rc<-test_imp$emi*12/(test_imp$ApplicantIncome+test_imp$CoapplicantIncome)
attach(train_imp)


test1 <- xgb.DMatrix(as.matrix(sapply(test_imp, as.numeric)))


#Predict on test data using the trained logit model
test_pred<- predict(bst, test1)
test_predval<-ifelse(test_pred>=0.5,"Y","N")

#Write the solution file
Final<-cbind.data.frame(test$Loan_ID,test_predval)
write.csv(Final,file="Solution File_V5.csv")





#Train a logistic regression model
model3<-glm(Loan_Status~Education+Self_Employed+rc+Credit_History+Property_Area,data=train,family = binomial(link="logit"),na.action = na.exclude,maxit=100)

#Predict training data responses using the above model
pred<-predict.glm(model3,newdata = NULL,type = c("response"))

#Set the classification boundary as 0.5
predval<-ifelse(pred>=0.5,"Y","N")

#Generate a confusion matrix
confusionMatrix(predval,Data$Loan_Status)


#########################XGB Parameter Tuning#######################################################


#bst <- xgboost(data = sparse_matrix, label = output_vector, max_depth = 5, min_child_weigt=1, gamma=
eta = 0.05, nthread = 2, nrounds = 2,objective = "binary:logistic", verbose = 1, eval.metric="auc")
