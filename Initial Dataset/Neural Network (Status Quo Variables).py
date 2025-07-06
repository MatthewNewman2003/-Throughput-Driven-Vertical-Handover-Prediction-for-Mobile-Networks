#Importing libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

#Reading CSV and converting it to a DataFrame
DataFrame = pd.read_csv("Final Status Quo Throughput Dataset.csv")

#Dropping rows with any missing values from DataFrame
DataFrame = DataFrame.dropna()

#Converting RAT Info into category codes
DataFrame["RAT Info"] = DataFrame["RAT Info"].astype("category")
DataFrame["RAT Info"] = DataFrame["RAT Info"].cat.codes

#Converting scenarios into category codes
DataFrame["Scenario"] = DataFrame["Scenario"].astype("category")
DataFrame["Scenario Category"] = DataFrame["Scenario"].cat.codes

#Outlining used dataset categories
UsedData = DataFrame[["GPS Long", "GPS Lat", "SS-RSRP", "SS-RSRQ", "SS-SINR", "Scenario Category", "RAT Info"]]

#Dividing the dataset into train and test datasets. 80% of the data is allocated to training, while the other 20% is set aside as testing/validation data
Train, TempTest = train_test_split(UsedData, test_size=0.2, random_state=100)

#Dividing the testing/validation data into 50% testing data and 50% validation data
Test, Validation = train_test_split(TempTest, test_size=0.5, random_state=100)

#Removing the RAT Info column from the train, test and validation datasets
TrainLabels = Train.pop("RAT Info")
TestLabels = Test.pop("RAT Info")
ValidationLabels = Validation.pop("RAT Info")

#Declaring MinMaxScaler to normalise the data
Scaler = RobustScaler()

#Normalising train, test and validation datasets
NormalisedTrainData = Scaler.fit_transform(Train)
NormalisedTestData = Scaler.transform(Test)
NormalisedValidationData = Scaler.transform(Validation)

#Declaring neural network
NeuralNetwork = Sequential()

#Adding a dense input layer, three dense hidden layers with increasing numbers of neurons for learning, and a dense output layer
NeuralNetwork.add(Dense(32, input_shape=(NormalisedTrainData.shape[1],)))
NeuralNetwork.add(Dense(32, activation="tanh"))
NeuralNetwork.add(Dense(64, activation="tanh"))
NeuralNetwork.add(Dense(128, activation="tanh"))
NeuralNetwork.add(Dense(1, activation="sigmoid"))

#Setting the hyperparameters of the model
LearningRate = 0.01
Optimiser = optimizers.SGD(LearningRate)
NeuralNetwork.compile(loss="binary_crossentropy", optimizer=Optimiser, metrics=["acc"])
BatchSize = 16

#Fitting the neural network to the training data
NeuralNetwork.fit(NormalisedTrainData, TrainLabels, batch_size=BatchSize, epochs=100, verbose=2, shuffle=True, validation_data = (NormalisedValidationData, ValidationLabels))

#Evaluating the neural network on the testing data
NeuralNetwork.evaluate(NormalisedTestData, TestLabels, verbose=2)

#Making predictions on the testing data
Prediction = NeuralNetwork.predict(NormalisedTestData, batch_size=BatchSize, verbose=2)
PredictionBoolean = np.round(Prediction, 0)

#Printing classification report to summarise the efficacy of the model
print(metrics.classification_report(TestLabels, PredictionBoolean))

#Saving model and RobustScaler to separate files
NeuralNetwork.save("NeuralNetworkStatusQuoVariables.h5")
pickle.dump(Scaler, open("NeuralNetworkScalerStatusQuoVariables.sav", "wb"))

#Notifying the user that K-fold cross validation is beginning
print("Beginning K-fold cross validation")

#Subroutine to create neural network for K-fold cross validation
def CreateModel():
    #Declaring neural network
    KFoldNeuralNetwork = Sequential()

    #Adding a dense input layer, three dense hidden layers with increasing numbers of neurons for learning, and a dense output layer
    KFoldNeuralNetwork.add(Dense(32, input_shape=(NormalisedData.shape[1],)))
    KFoldNeuralNetwork.add(Dense(32, activation="tanh"))
    KFoldNeuralNetwork.add(Dense(64, activation="tanh"))
    KFoldNeuralNetwork.add(Dense(128, activation="tanh"))
    KFoldNeuralNetwork.add(Dense(1, activation="sigmoid"))

    #Setting the hyperparameters of the model
    LearningRate = 0.01
    Optimiser = optimizers.SGD(LearningRate)
    KFoldNeuralNetwork.compile(loss="binary_crossentropy", optimizer=Optimiser, metrics=["acc"])
    BatchSize = 16
    #Returning the neural network at the end of the subroutine
    return KFoldNeuralNetwork

#Outlining X and Y for K-fold cross validation
X = DataFrame[["GPS Long", "GPS Lat", "SS-RSRP", "SS-RSRQ", "SS-SINR", "Scenario Category", "RAT Info"]]
Y = X.pop("RAT Info")

#Declaring RobustScaler to normalise the data for K-fold cross validation
KFoldScaler = RobustScaler()

#Normalising data for K-fold cross validation
NormalisedData = KFoldScaler.fit_transform(X)

#Defining cross-validation metrics
Metrics = {"accuracy" : "accuracy",
           "precision" : "precision_macro",
           "recall" : "recall_macro"}

#Creating KerasClassifier wrapper for neural network for K-fold cross validation, performing K-fold cross validation and showing the results
NeuralNetworkSKLearn = KerasClassifier(model=CreateModel, epochs=100, batch_size=16, verbose=2)
Validator = KFold(n_splits=10, random_state=1, shuffle=True)
Results = cross_validate(NeuralNetworkSKLearn, NormalisedData, Y, scoring=Metrics, cv=Validator)
print(Results)
print("Accuracy:",np.mean(Results["test_accuracy"]))
print("Precision:",np.mean(Results["test_precision"]))
print("Recall:",np.mean(Results["test_recall"]))
