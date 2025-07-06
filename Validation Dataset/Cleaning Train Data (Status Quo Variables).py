#Importing Pandas
import pandas as pd

#Reading CSV and using relevant columns
TrainData = pd.read_csv("CombinedTrainDataset.csv", usecols=["Timestamp", "Longitude", "Latitude", "Speed", "NetworkMode", "RSRP", "RSRQ", "State"])

#Printing top rows of train dataset
print(TrainData.head())

#Turning the timestamp into more manipulable date and time columns
TrainData[["Date", "Time"]] = TrainData["Timestamp"].str.split("_", expand=True)

#Printing top rows of modified dataset
print(TrainData.head())

#Declaring array for scenario markers
ScenarioMarkers = []

#Adding a relevant scenario marker to each record in the dataset
for i in range(0, len(TrainData)):
    ScenarioMarkers.append("Train")

#Assigning a scenario marker to each record in the dataset
TrainDataWithScenario = TrainData.assign(Scenario=ScenarioMarkers)

#Printing top rows of final dataset
print(TrainDataWithScenario.head())

#Writing final dataset to a CSV file
TrainDataWithScenario.to_csv("FinalTrainDataset (Status Quo Variables).csv", index=False)
