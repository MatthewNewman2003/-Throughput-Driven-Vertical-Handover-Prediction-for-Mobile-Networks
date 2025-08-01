#Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import pickle

#Reading CSV and converting it to a DataFrame
DataFrame = pd.read_csv("Final Throughput Dataset.csv")

#Dropping rows with any missing values from DataFrame
DataFrame = DataFrame.dropna()

DataFrame["Scenario"] = DataFrame["Scenario"].astype("category")
DataFrame["Scenario Category"] = DataFrame["Scenario"].cat.codes

#Outlining x and y. x represents the independent predictor variables, while y represents the dependent target variable
X = DataFrame[["GPS Long", "GPS Lat", "Current Netw. DL", "Current Netw. UL", "Mean Netw. DL", "Mean Netw. UL", "Current Netw. DL Avg", "Current Netw. DL Max", "Current Netw. UL Avg", "Current Netw. UL Max", "Scenario Category"]]
Y = DataFrame["RAT Info"]

#Dividing the dataset into train and test datasets. 80% of the data is allocated to training, while the other 20% is set aside as testing data
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2, random_state=100)

#Generating random forest classification model
RandomForest = RandomForestClassifier(n_estimators=25, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", max_depth=27)

#Searching for optimal random forest hyperparameters for the training data given
RandomForest.fit(XTrain, YTrain)

#Making predictions on the test dataset using the optimised random forest model
YPrediction = RandomForest.predict(XTest)

#Calculating model accuracy
Accuracy = accuracy_score(YTest, YPrediction)
FiveGPrecision = precision_score(YTest, YPrediction, pos_label="5G EN-DC")
FiveGRecall = recall_score(YTest, YPrediction, pos_label="5G EN-DC")
FiveGFOneScore = f1_score(YTest, YPrediction, pos_label="5G EN-DC")
FourGPrecision = precision_score(YTest, YPrediction, pos_label="LTE")
FourGRecall = recall_score(YTest, YPrediction, pos_label="LTE")
FourGFOneScore = f1_score(YTest, YPrediction, pos_label="LTE")

#Outputting model accuracy
print("Model accuracy on test data is:",Accuracy)
print("5G Precision:",FiveGPrecision)
print("5G Recall:",FiveGRecall)
print("5G F1 Score:",FiveGFOneScore)
print("4G Precision:",FourGPrecision)
print("4G Recall:",FourGRecall)
print("4G F1 Score:",FourGFOneScore)

#Saving the trained model to a separate file
#pickle.dump(RandomForest, open("RandomForestDLAndUL.sav", "wb"))

#Telling the user that K-fold cross validation is beginning
print("Beginning K-fold cross validation.")

#Generating random forest classification model for K-fold cross validation
KFoldRandomForest = RandomForestClassifier(n_estimators=25, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", max_depth=27)

#Creating K-fold cross validation validator
Validator = KFold(n_splits=10, random_state=1, shuffle=True)

#Defining cross-validation metrics
Metrics = {"accuracy" : "accuracy",
           "precision" : "precision_macro",
           "recall" : "recall_macro"}

#Performing K-fold cross validation on model
Scores = cross_validate(KFoldRandomForest, X, Y, scoring=Metrics, cv=Validator, n_jobs=-1)

#Outputting the accuracy gained from K-fold cross validation
print(Scores)
print("Accuracy:",np.mean(Scores["test_accuracy"]))
print("Precision:",np.mean(Scores["test_precision"]))
print("Recall:",np.mean(Scores["test_recall"]))
