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
DataFrame = pd.read_csv("Filtered Combined Dataset (Status Quo Variables).csv", na_values="-")

#Dropping rows with any missing values from DataFrame
DataFrame = DataFrame.dropna()

#Converting Scenario into a Category type and encoding the values as integers
DataFrame["Scenario"] = DataFrame["Scenario"].astype("category")
DataFrame["Scenario Category"] = DataFrame["Scenario"].cat.codes

#Outlining x and y. x represents the independent predictor variables, while y represents the dependent target variable
X = DataFrame[["Longitude", "Latitude", "RSRP", "RSRQ", "Scenario Category"]]
Y = DataFrame["NetworkMode"]

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

#Calculating model accuracy
Accuracy = accuracy_score(YTest, YPrediction)
LTEPrecision = precision_score(YTest, YPrediction, average="macro", labels=["LTE"])
LTERecall = recall_score(YTest, YPrediction, average="macro", labels=["LTE"])
LTEFOneScore = f1_score(YTest, YPrediction, average="macro", labels=["LTE"])
HSPAPlusPrecision = precision_score(YTest, YPrediction, average="macro", labels=["HSPA+"])
HSPAPlusRecall = recall_score(YTest, YPrediction, average="macro", labels=["HSPA+"])
HSPAPlusFOneScore = f1_score(YTest, YPrediction, average="macro", labels=["HSPA+"])

#Outputting model accuracy
print("Model accuracy on test data is:",Accuracy)
print("LTE Precision:",LTEPrecision)
print("LTE Recall:",LTERecall)
print("LTE F1 Score:",LTEFOneScore)
print("HSPA+ Precision:",HSPAPlusPrecision)
print("HSPA+ Recall:",HSPAPlusRecall)
print("HSPA+ F1 Score:",HSPAPlusFOneScore)

#Saving the trained model and RobustScaler to separate files
pickle.dump(NaiveBayesClassifier, open("NaiveBayesClassifierStatusQuoVariables.sav", "wb"))
pickle.dump(Scaler, open("RobustScalerStatusQuoVariables.sav", "wb"))

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
