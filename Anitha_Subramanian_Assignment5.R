library(caret)
data(scat)
dataset<-scat

#1.Set the species column as outcome/target and converting to numeric
dataset$Species<-as.numeric(factor(dataset$Species))

#2. Remove the Month, Year, Site, Location features
dataset<-dataset[c(1,6:19)]

#3. Check if any values are null. If there are, impute missing values using KNN
sum(is.na(dataset))
preProcValues <- preProcess(dataset, method = c("knnImpute","center","scale"))
library('RANN')
dataset_processed <- predict(preProcValues, dataset)
sum(is.na(dataset_processed))

#4. Converting every categorical variable to numerical (if needed)
dmy <- dummyVars(" ~ .", data = dataset_processed,fullRank = T)
dataset_transformed <- data.frame(predict(dmy, newdata = dataset_processed))
str(dataset_transformed)
dataset_transformed$Species<-as.factor(dataset_transformed$Species)

#5. With a seed of 100, 75% training, 25% testing. Build the following models: randomforest, neural net, naive bayes and GBM
#Splitting dataset into 75% training and 25% testing
set.seed(100)
index <- createDataPartition(dataset_transformed$Species, p=0.75, list=FALSE)
trainSet <- dataset_transformed[index,]
testSet <- dataset_transformed[-index,]
str(trainSet)
#Feature Selection
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Species'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
Animal_Pred <- rfe(trainSet[,predictors], trainSet[,outcomeName],
                         rfeControl = control)
Animal_Pred
predictors<-c("CN", "d13C", "d15N", "Mass")
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')
model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet')
install.packages("klaR")
install.packages("promises")
install.packages("mime")
install.packages("fastmap")
library(klaR)
model_nb<-train(trainSet[,predictors],trainSet[,outcomeName],method='naive_bayes')

#5.a Summarizing the models
print(model_gbm)
print(model_rf)
print(model_nnet)
print(model_nb)

#5.b plot variable of importance, for the predictions (use the prediction set) display
library(gbm)
varImp(object=model_gbm)
plot(varImp(object=model_gbm),main="GBM - Variable Importance")
varImp(object=model_rf)
plot(varImp(object=model_rf),main="Random Forest - Variable Importance")
varImp(object=model_nnet)
varImp(object=model_nb)
plot(varImp(object=model_nb),main="Naive Bayes - Variable Importance")

#5.c Confusion Matrix
#Predictions
predictions<-predict.train(object=model_gbm,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])
predictions<-predict.train(object=model_rf,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])
predictions<-predict.train(object=model_nnet,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])
predictions<-predict.train(object=model_nb,testSet[,predictors],type="raw")
table(predictions)
confusionMatrix(predictions,testSet[,outcomeName])

#6.For the BEST performing models of each (randomforest, neural net, naive bayes and gbm) create and display a data frame that has the following columns: ExperimentName, accuracy, kappa. Sort the data frame by accuracy.
print(model_gbm$resample[order(-model_gbm$resample$Accuracy),])
print(model_rf$resample[order(-model_rf$resample$Accuracy),])
print(model_nnet$resample[order(-model_nnet$resample$Accuracy),])
print(model_nb$resample[order(-model_nb$resample$Accuracy),])

#7.Tune the GBM model using tune length = 20
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=20)

#7.a) print the model summary
print(model_gbm)

#7.b) plot the models
plot(model_gbm)

#8.Using GGplot and gridExtra to plot all variable of importance plots into one single plot.
install.packages("gridExtra")
library(gridExtra)
library(grid)
g1<-ggplot(varImp(object=model_gbm),main="GBM - Variable Importance")
g2<-ggplot(varImp(object=model_rf),main="Random Forest - Variable Importance")
grid.arrange(g1,g2,ncol=2)

#9.Which model performs the best? and why do you think this is the case? Can we accurately predict species on this dataset?
#Answer - From the accuary of the results from the 4 models (gbm,random forest,neural net and naive bayes) gbm works best as it is able to predict 93% accurately of the Species.

#10.a.Using feature selection with rfe in caret and the repeatedcv method: Find the top 3 predictors and build the same models as in 6 and 8 with the same parameters
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Species'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
Animal_Pred <- rfe(trainSet[,predictors], trainSet[,outcomeName],
                   rfeControl = control)
Animal_Pred
predictors1<-c("CN", "d13C", "d15N")
modelgbm<-train(trainSet[,predictors1],trainSet[,outcomeName],method='gbm')
modelrf<-train(trainSet[,predictors1],trainSet[,outcomeName],method='rf')
modelnnet<-train(trainSet[,predictors1],trainSet[,outcomeName],method='nnet')
install.packages("klaR")
install.packages("promises")
install.packages("mime")
install.packages("fastmap")
library(klaR)
modelnb<-train(trainSet[,predictors1],trainSet[,outcomeName],method='naive_bayes')

print(modelgbm$resample[order(-modelgbm$resample$Accuracy),])
print(modelrf$resample[order(-modelrf$resample$Accuracy),])
print(modelnnet$resample[order(-modelnnet$resample$Accuracy),])
print(modelnb$resample[order(-modelnb$resample$Accuracy),])

g1<-ggplot(varImp(object=modelgbm),main="GBM - Variable Importance")
g2<-ggplot(varImp(object=modelrf),main="Random Forest - Variable Importance")
grid.arrange(g1,g2,ncol=2)

#10.b.Create a dataframe that compares the non-feature selected models (the same as on 7)and add the best BEST performing models of each (randomforest, neural net, naive bayes and gbm) and display the data frame that has the following columns: ExperimentName, accuracy, kappa. Sort the data frame by accuracy.
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)
#Creating dataframe
modelgbm<-train(trainSet[,predictors1],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=20)
modelrf<-train(trainSet[,predictors1],trainSet[,outcomeName],method='rf',trControl=fitControl,tuneLength=20)
modelnnet<-train(trainSet[,predictors1],trainSet[,outcomeName],method='nnet',trControl=fitControl,tuneLength=20)
modelnb<-train(trainSet[,predictors1],trainSet[,outcomeName],method='nb',trControl=fitControl,tuneLength=20)

print(modelgbm$resample[order(-modelgbm$resample$Accuracy),])
print(modelrf$resample[order(-modelrf$resample$Accuracy),])
print(modelnnet$resample[order(-modelnnet$resample$Accuracy),])
print(modelnb$resample[order(-modelnb$resample$Accuracy),])

#10.c. Which model performs the best? and why do you think this is the case? Can we accurately predict species on this dataset?
#Answer - By taking only top 3 predictors, We could see from the accuary of the results of the 4 models (gbm,random forest,neural net and naive bayes) neural net works best as it is able to predict 94% accurately of the Species.


install.packages("knitr")
library(knitr)
getwd()
knitr::stitch('Anitha_Subramanian_Assignment5.r')
