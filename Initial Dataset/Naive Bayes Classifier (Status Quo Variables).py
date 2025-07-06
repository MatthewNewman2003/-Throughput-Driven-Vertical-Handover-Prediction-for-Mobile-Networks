#Importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
import pickle

#Reading CSV and converting it to a DataFrame
DataFrame = pd.read_csv("Final Status Quo Throughput Dataset.csv")

#Dropping rows with any missing values from DataFrame
DataFrame = DataFrame.dropna()

#Converting Scenario into a Category type and encoding the values as integers
DataFrame["Scenario"] = DataFrame["Scenario"].astype("category")
DataFrame["Scenario Category"] = DataFrame["Scenario"].cat.codes

#Outlining x and y. x represents the independent predictor variables, while y represents the dependent target variable
X = DataFrame[["GPS Long", "GPS Lat", "SS-RSRP", "SS-RSRQ", "SS-SINR", "Scenario Category"]]
Y = DataFrame["RAT Info"]

#Dividing the dataset into train and test datasets. 80% of the data is allocated to training, while the other 20% is set aside as testing data
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2, random_state=100)

#Denoting the columns of the dataset
Columns = XTrain.columns

#Declaring a scaler to scale the data for input into a Naive-Bayes Classifier
Scaler = RobustScaler()

#Fitting the scaler to the training dataset and transforming it
XTrain = Scaler.fit_transform(XTrain)

#Transforming the testing dataset
XTest = Scaler.transform(XTest)

#Creating a Pandas DataFrame out of the scaled training data
XTrain = pd.DataFrame(XTrain, columns=[Columns])

#Creating a Pandas DataFrame out of the scaled testing data
XTest = pd.DataFrame(XTest, columns=[Columns])

#Declaring a Naive-Bayes Classifier
NaiveBayesClassifier = GaussianNB()

#Fitting the Naive-Bayes Classifier to the training and testing datasets
NaiveBayesClassifier.fit(XTrain, YTrain)

#Making predictions on the testing data using the Naive-Bayes Classifier
YPrediction = NaiveBayesClassifier.predict(XTest)

#Calculating the accuracy of the model's predictions
Accuracy = accuracy_score(YTest, YPrediction)
FiveGPrecision = precision_score(YTest, YPrediction, pos_label="5G EN-DC")
FiveGRecall = recall_score(YTest, YPrediction, pos_label="5G EN-DC")
FiveGFOneScore = f1_score(YTest, YPrediction, pos_label="5G EN-DC")
FourGPrecision = precision_score(YTest, YPrediction, pos_label="LTE")
FourGRecall = recall_score(YTest, YPrediction, pos_label="LTE")
FourGFOneScore = f1_score(YTest, YPrediction, pos_label="LTE")

#Outputting the model's prediction accuracy
print("Accuracy on test dataset:",Accuracy)
print("5G Precision:",FiveGPrecision)
print("5G Recall:",FiveGRecall)
print("5G F1 Score:",FiveGFOneScore)
print("4G Precision:",FourGPrecision)
print("4G Recall:",FourGRecall)
print("4G F1 Score:",FourGFOneScore)

#Saving the trained model and scaler to separate files
pickle.dump(NaiveBayesClassifier, open("NaiveBayesClassifierStatusQuo.sav", "wb"))
pickle.dump(Scaler, open("RobustScalerStatusQuo.sav", "wb"))

#Telling the user that K-fold cross validation is beginning
print("Beginning K-fold cross validation.")

#Denoting the columns of the dataset for K-fold cross validation
KFoldColumns = X.columns

#Declaring a scaler to scale the data for input into a Naive-Bayes Classifier for K-fold cross validation
KFoldScaler = RobustScaler()

#Fitting the scaler to the training dataset and transforming it for K-fold cross validation
X = KFoldScaler.fit_transform(X)

#Creating a Pandas DataFrame out of the scaled training data for K-fold cross validation
X = pd.DataFrame(X, columns=[KFoldColumns])

#Declaring a Naive-Bayes Classifier for K-fold cross validation
KFoldNaiveBayesClassifier = GaussianNB()

#Creating K-fold cross validation validator
Validator = KFold(n_splits=10, random_state=1, shuffle=True)

#Defining cross-validation metrics
Metrics = {"accuracy" : "accuracy",
           "precision" : "precision_macro",
           "recall" : "recall_macro"}

#Performing K-fold cross validation on model
Scores = cross_validate(KFoldNaiveBayesClassifier, X, Y, scoring=Metrics, cv=Validator, n_jobs=-1)

#Outputting the accuracy gained from K-fold cross validation
print(Scores)
print("Accuracy:",np.mean(Scores["test_accuracy"]))
print("Precision:",np.mean(Scores["test_precision"]))
print("Recall:",np.mean(Scores["test_recall"]))
