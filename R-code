#Remove everything in the environment
rm(list=ls())

#Load libraries and data
load.libraries <- c('ggplot2','stringr','Matrix','glmnet','xgboost','randomForest','Metrics','dplyr','caret','scales','e1071','corrplot','onehot','rpart')
install.lib <- load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dependences = TRUE)
sapply(load.libraries, require, character = TRUE)
train<- fread('https://storage.googleapis.com/kaggle-competitions-data/kaggle/5407/train.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1518158777&Signature=EHvARB7aAauIi6aEJXocPcXcyoX6yqT2FxGvbj0NMNCGsDqPQBYSMJwlh9%2FHNom9wVJF8HFm01HDiKf0CnIRmwX6JPZFsUUFII1%2FI87EbQHajWRZ22vt%2BBeHDMIy1oEBcwi3DuuXEmmftg9V3XQmxjb%2BNfwj5OD7MmxavTPyACGlCgJ0wInUtulRRb%2BAcJVH%2BAhTLX1gKUf47zCcE%2FbkB%2BlZSUUaDVvbpZIVKGx9hX1%2FRMuzN73NRTDOBx6ZhWZEPQ5kWLlt2pPyBrlVf%2FsHbIL%2FpH3pCUmnPaAPlfl5aGL2YTcm3ImO1tEDMX78I3kWZZXymAHcqetIzrL5lzyt7w%3D%3D', stringsAsFactors = FALSE)
test<-fread('https://storage.googleapis.com/kaggle-competitions-data/kaggle/5407/test.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1518159087&Signature=B0i%2BgEbdwMzLylANTqRj3rNKRi7%2FBA2xX4Cpf9fCQA3y8R5S2hEGmRiCvjBUOu%2FyWFIEb4uBm0nSGt7oJ8G%2FCX%2FhbMAbDiprS5t6eSTNJ8nQzhdEsI1sBpAJ3IJmfgmPm4emjlTEXC6GIS1y2egbWmEnVbTkfr8ZZENTMSb9TiWe6nzTs8Mn2%2BteSP61RGv9xfoyH5KZDCt4GiVYTfgJcOamsPe%2Fi6zbmkYlQiR0ELSCCJLqu9ub%2F4TPYcvJbZs%2BTp%2FxZEpi5wkY0Rfnp7cggZXBMB04zxu5Y%2BH45kOnjEbHVqQY%2FIDEKXVmDQFy%2BCJ0RIpaDVOS%2FPV8G5nSJ2jVKQ%3D%3D', stringsAsFactors = FALSE)
dim(train)
dim(test)

# combine the datasets
df.combined <- rbind(within(train, rm('Id','SalePrice')), within(test, rm('Id')))
dim(df.combined)

na.cols <- colSums(is.na(df.combined))
na.cols<-na.cols[na.cols>0]
sort(na.cols, decreasing = TRUE)
paste('There are', length(na.cols), 'columns with missing values')

# helper function for plotting categoric data for easier data visualization
plot.categoric <- function(cols, df){
  for (col in cols) {
    order.cols <- names(sort(table(df.combined[,col]), decreasing = TRUE))
    
    num.plot <- qplot(df[,col]) +
      geom_bar(fill = 'cornflowerblue') +
      geom_text(aes(label = ..count..), stat='count', vjust=-0.5) +
      theme_minimal() +
      scale_y_continuous(limits = c(0,max(table(df[,col]))*1.1)) +
      scale_x_discrete(limits = order.cols) +
      xlab(col) +
      theme(axis.text.x = element_text(angle = 30, size=12))
    
    print(num.plot)
  }
}


#Plot PoolQC
plot.categoric('PoolQC', df.combined)
#Houses with PoolArea>0 and PoolQC="NA"
df.combined[(df.combined$PoolArea>0)&is.na(df.combined$PoolQC),c('PoolQC','PoolArea')]

df.combined[,c('PoolQC','PoolArea')] %>%
  group_by(PoolQC) %>%
  summarise(mean=mean(PoolArea),counts=n())

df.combined$PoolQC[2421]='Ex'
df.combined$PoolQC[2504]='Ex'
df.combined$PoolQC[2600]='Fa'

df.combined$PoolQC[is.na(df.combined$PoolQC)]<-'None'

#Columns related to Garage
garage.cols <- c('GarageYrBlt','GarageArea', 'GarageCars', 'GarageQual', 'GarageFinish', 'GarageCond', 'GarageType')
garage.na.cols <- colSums(is.na(df.combined[,c('GarageYrBlt','GarageArea', 'GarageCars', 'GarageQual', 'GarageFinish', 'GarageCond', 'GarageType')]))
garage.na.cols<-garage.na.cols[garage.na.cols>0]
garage.na.cols

#Treat GarageYrBlt
length(which(df.combined$GarageYrBlt == df.combined$YearBuilt))

idx <- which(is.na(df.combined$GarageYrBlt))
df.combined[idx, 'GarageYrBlt'] <- df.combined[idx, 'YearBuilt']


#Treat GarageArea and GarageCars
which(is.na(df.combined$GarageArea))
which(is.na(df.combined$GarageCars))
##Since both are NAs in the same row, will replece both with 0
df.combined$GarageArea[2577]<-0
df.combined$GarageCars[2577]<-0

#Treat GarageType
df.combined[,c('GarageArea','GarageType')] %>%
  group_by(GarageType) %>%
  summarise(mean=mean(GarageArea,na.rm=TRUE),counts=n())

df.combined$GarageType[is.na(df.combined$GarageType)]<-'NoGarage'

#Treat GarageQual, GarageFinish, GarageCond together
df.g<-df.combined[,c('GarageQual','GarageFinish','GarageCond','GarageArea')]
df.g[is.na(df.g$GarageQual)&is.na(df.g$GarageFinish)&is.na(df.g$GarageCond)&(df.g$GarageArea==0),]

df.combined$GarageQual[is.na(df.combined$GarageQual)&is.na(df.combined$GarageFinish)&is.na(df.combined$GarageCond)&(df.combined$GarageArea==0)]<-'NoGarage'
df.combined$GarageFinish[is.na(df.combined$GarageFinish)&is.na(df.combined$GarageCond)&(df.combined$GarageArea==0)]<-'NoGarage'
df.combined$GarageCond[is.na(df.combined$GarageCond)&(df.combined$GarageArea==0)]<-'NoGarage'

df.combined[is.na(df.combined$GarageQual)&is.na(df.combined$GarageFinish)&is.na(df.combined$GarageCond)&(df.combined$GarageArea>0),'GarageArea']

df.combined[,c('GarageArea','GarageQual')] %>%
  group_by(GarageQual) %>%
  summarise(mean=mean(GarageArea,na.rm=TRUE),counts=n())

df.combined[,c('GarageArea','GarageFinish')] %>%
  group_by(GarageFinish) %>%
  summarise(mean=mean(GarageArea,na.rm=TRUE),counts=n())

df.combined[,c('GarageArea','GarageCond')] %>%
  group_by(GarageCond) %>%
  summarise(mean=mean(GarageArea,na.rm=TRUE),counts=n())

df.combined$GarageQual[is.na(df.combined$GarageQual)&is.na(df.combined$GarageFinish)&is.na(df.combined$GarageCond)&(df.combined$GarageArea>0)]<-'Fa'
df.combined$GarageFinish[is.na(df.combined$GarageFinish)&is.na(df.combined$GarageCond)&(df.combined$GarageArea>0)]<-'Unf'
df.combined$GarageCond[is.na(df.combined$GarageCond)&(df.combined$GarageArea>0)]<-'Fa'


#Columns related to Bsmt(Basement)
Bsmt.cols <- c('BsmtCond','BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'BsmtFullBath', 'BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF')
Bsmt.na.cols <- colSums(is.na(df.combined[,c('BsmtCond','BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'BsmtFullBath', 'BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF')]))
Bsmt.na.cols<-Bsmt.na.cols[Bsmt.na.cols>0]
Bsmt.na.cols

#Treat BsmtCond
which(is.na(df.combined$BsmtCond)&(df.combined$TotalBsmtSF>0))

df.combined[is.na(df.combined$BsmtCond)&(df.combined$TotalBsmtSF>0),c('TotalBsmtSF','BsmtCond')]
df.combined[,c('TotalBsmtSF','BsmtCond')] %>%
  group_by(BsmtCond) %>%
  summarise(mean=mean(TotalBsmtSF),counts=n())

df.combined$BsmtCond[is.na( df.combined$BsmtCond)&(df.combined$TotalBsmtSF==1426)]<-'Gd'
df.combined$BsmtCond[is.na( df.combined$BsmtCond)&(df.combined$TotalBsmtSF==1127)]<-'Gd'
df.combined$BsmtCond[is.na( df.combined$BsmtCond)&(df.combined$TotalBsmtSF==995)]<-'Po'
df.combined$BsmtCond[is.na( df.combined$BsmtCond)&(df.combined$TotalBsmtSF==0)]<-'NoBsmt'

#Treat BsmtExposure, BsmtQual, BsmtFinType2 and BsmtFinType1 together
dim(df.combined[is.na(df.combined$BsmtExposure)&is.na(df.combined$BsmtQual)&is.na(df.combined$BsmtFinType1)&is.na(df.combined$BsmtFinType2),c('BsmtExposure', 'BsmtQual', 'BsmtFinType2','BsmtFinType1')])

dim(df.combined[is.na(df.combined$BsmtExposure)&is.na(df.combined$BsmtQual)&is.na(df.combined$BsmtFinType1)&is.na(df.combined$BsmtFinType2)&(df.combined$TotalBsmtSF==0),c('BsmtExposure', 'BsmtQual', 'BsmtFinType2','BsmtFinType1')])
df.combined$BsmtExposure[is.na(df.combined$BsmtExposure)&(df.combined$TotalBsmtSF==0)]<-'NoBsmt'
df.combined$BsmtQual[is.na(df.combined$BsmtQual)&(df.combined$TotalBsmtSF==0)]<-'NoBsmt'
df.combined$BsmtFinType1[is.na(df.combined$BsmtFinType1)&(df.combined$TotalBsmtSF==0)]<-'NoBsmt'
df.combined$BsmtFinType2[is.na(df.combined$BsmtFinType2)&(df.combined$TotalBsmtSF==0)]<-'NoBsmt'

which(is.na(df.combined$BsmtExposure)&(df.combined$TotalBsmtSF>0))
which(is.na(df.combined$BsmtQual)&(df.combined$TotalBsmtSF>0))
which(is.na(df.combined$BsmtFinType1)&(df.combined$TotalBsmtSF>0))
which(is.na(df.combined$BsmtFinType2)&(df.combined$TotalBsmtSF>0))

xtabs(~BsmtQual+BsmtExposure,df.combined)

df.combined$BsmtExposure[949]<-'No'
df.combined$BsmtExposure[1488]<-'No'
df.combined$BsmtExposure[2349]<-'No'

xtabs(~BsmtCond+BsmtQual,df.combined)
df.combined$BsmtQual[2218:2219]<-'TA'

xtabs(~BsmtCond+BsmtFinType2,df.combined)
df.combined$BsmtFinType2[333]<-'Unf'

df.combined[2121, c('BsmtCond','BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1')]<-'NoBsmt'
df.combined[2121, c('BsmtFullBath', 'BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF')]<-0
df.combined[2189,c('BsmtFullBath','BsmtHalfBath')]<-0

#Treat MiscFeature
df.combined$MiscFeature[is.na(df.combined$MiscFeature)]<-'None'

#Treat Fence
df.combined$Fence[is.na(df.combined$Fence)]<-'None'

#Treat Alley
df.combined$Alley[is.na(df.combined$Alley)]<-'None'

#Treat FireplaceQu
df.combined[is.na(df.combined$FireplaceQu)&(df.combined$Fireplaces>0),c('FireplaceQu','Fireplaces')]
df.combined$FireplaceQu[is.na(df.combined$FireplaceQu)]<-'None'

#Treat MasVnrType and MasVnrArea
dim(df.combined[is.na(df.combined$MasVnrArea)&is.na(df.combined$MasVnrType),])
df.combined$MasVnrArea[is.na(df.combined$MasVnrArea)]<-0
df.combined$MasVnrType[is.na(df.combined$MasVnrType)]<-'None'

#Treat LotFrontage
lot.by.nbrh <- df.combined[,c('Neighborhood','LotFrontage')] %>%
  group_by(Neighborhood) %>%
  summarise(median = median(LotFrontage, na.rm = TRUE))
lot.by.nbrh    

idx<-which(is.na(df.combined$LotFrontage))

for (i in idx){
  lot.median <- lot.by.nbrh[lot.by.nbrh == df.combined$Neighborhood[i],'median']
  df.combined[i,'LotFrontage'] <- lot.median[[1]]
}

#Treat MSZoning
df.combined[is.na(df.combined$MSZoning),c('MSZoning','MSSubClass')]

table(df.combined$MSZoning, df.combined$MSSubClass)

df.combined$MSZoning[c(2217, 2905)] = 'RL'
df.combined$MSZoning[c(1916, 2251)] = 'RM'

#Treat Utilities, Functional, Electrical, KitchenQual, SaleType, Exterior1st and Exterior2nd with mode
df.combined$Utilities[is.na(df.combined$Utilities)]<-mode(df.combined$Utilities)

df.combined$Functional[is.na(df.combined$Functional)]<-mode(df.combined$Functional)

df.combined$Electrical[is.na(df.combined$Electrical)]<-mode(df.combined$Electrical)

df.combined$KitchenQual[is.na(df.combined$KitchenQual)]<-mode(df.combined$KitchenQual)

df.combined$SaleType[is.na(df.combined$SaleType)]<-mode(df.combined$SaleType)

df.combined$Exterior1st[is.na(df.combined$Exterior1st)]<-mode(df.combined$Exterior1st)

df.combined$Exterior2nd[is.na(df.combined$Exterior2nd)]<-mode(df.combined$Exterior2nd)


###########All Missing values are imputed#########################################################
#cat_features<-names(which(sapply(df.combined,is.character)))
#num_features<-names(which(sapply(df.combined,is.numeric)))

group.df <- df.combined[1:1460,]
group.df$SalePrice<- train$SalePrice
#df.train<-onehot(group.df,stringsAsFactors = TRUE,addNA = FALSE,max_levels = 25)

############ One-hot emcoding done###################

sum(sapply(group.df_factor, is.character)) # 2

group.df_factor <- group.df %>%
  mutate_if(sapply(group.df, is.character), as.factor)
 
setnames(group.df_factor,c('1stFlrSF','2ndFlrSF','3SsnPorch'),c('FFlrSF','SFlrSF','SsnPorch'))
data.tr<-cbind(group.df$SalePrice,group.df_factor)

control.parm<-rpart.control(minsplit = 30,cp = 6.051931e-04)

dt.fit<-rpart(group.df$SalePrice~.,data=data.tr,method = "anova",control = control.parm)

plotcp(dt.fit)
x<-printcp(dt.fit)
x[,4]+x[,5]

# plot tree 
plot(pfit, uniform=TRUE, 
     main="Regression Tree House Loans ")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)

# prune the tree 
pfit<- prune(dt.fit, cp=0.03) # from cptable   

pred<-predict(pfit)

rmse_eval <- function(y.true, y.pred) {
  mse_eval <- sum((y.true - exp(y.pred)-1)^2) / length(y.true)
  return(sqrt(mse_eval))
}

rmse_eval(as.numeric(group.df$SalePrice),pred)
install.packages("RGtk2", depen=T)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

fancyRpartPlot(dt.fit)
