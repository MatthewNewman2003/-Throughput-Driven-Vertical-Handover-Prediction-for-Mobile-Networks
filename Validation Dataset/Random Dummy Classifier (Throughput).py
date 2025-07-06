#Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.metrics import *

#Reading CSV and converting it to a DataFrame
DataFrame = pd.read_csv("Filtered Combined Dataset (Throughput).csv")

#Dropping rows with any missing values from DataFrame
DataFrame = DataFrame.dropna()

#Converting State into a Category type and encoding the values as integers
DataFrame["State"] = DataFrame["State"].astype("category")
DataFrame["State Category"] = DataFrame["State"].cat.codes

#Converting Scenario into a Category type and encoding the values as integers
DataFrame["Scenario"] = DataFrame["Scenario"].astype("category")
DataFrame["Scenario Category"] = DataFrame["Scenario"].cat.codes

#Outlining x and y. x represents the independent predictor variables, while y represents the dependent target variable
X = DataFrame[["Longitude", "Latitude", "DL_bitrate", "UL_bitrate", "Scenario Category"]]
Y = DataFrame["NetworkMode"]

#Dividing the dataset into train and test datasets. 80% of the data is allocated to training, while the other 20% is set aside as testing data
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2, random_state=100)

#Generating dummy classification model
Dummy = DummyClassifier(strategy="uniform")

#Fitting the dummy classifier to the training data
Dummy.fit(XTrain, YTrain)

#Making predictions on the test dataset using the dummy model
YPrediction = Dummy.predict(XTest)

#Calculating model accuracy
Accuracy = accuracy_score(YTest, YPrediction)
ThreeGPrecision = precision_score(YTest, YPrediction, pos_label="HSPA+")
ThreeGRecall = recall_score(YTest, YPrediction, pos_label="HSPA+")
FourGPrecision = precision_score(YTest, YPrediction, pos_label="LTE")
FourGRecall = recall_score(YTest, YPrediction, pos_label="LTE")
PrecisionArray = [ThreeGPrecision, FourGPrecision]
RecallArray = [ThreeGRecall, FourGRecall]
Precision = np.mean(PrecisionArray)
Recall = np.mean(RecallArray)

#Outputting model accuracy
print("Model accuracy on test data is:",Accuracy)
print("Precision:",Precision)
print("Recall:",Recall)

#Notifying the user that K-fold cross validation is beginning
print("Beginning K-fold cross validation")

#Creating DummyClassifier for K-fold cross validation
KFoldDummy = DummyClassifier(strategy="uniform")

#Creating K-fold cross validation validator
Validator = KFold(n_splits=10, random_state=1, shuffle=True)

#Defining cross-validation metrics
Metrics = {"accuracy" : "accuracy",
           "precision" : "precision_macro",
           "recall" : "recall_macro"}

#Performing K-fold cross validation on model
Scores = cross_validate(KFoldDummy, X, Y, scoring=Metrics, cv=Validator, n_jobs=-1)

#Outputting the accuracy gained from K-fold cross validation
print(Scores)
print("Accuracy:",np.mean(Scores["test_accuracy"]))
print("Precision:",np.mean(Scores["test_precision"]))
print("Recall:",np.mean(Scores["test_recall"]))
