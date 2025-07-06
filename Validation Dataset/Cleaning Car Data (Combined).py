#Importing Pandas
import pandas as pd

#Reading CSV and using relevant columns
CarData = pd.read_csv("CombinedCarDataset.csv", usecols=["Timestamp", "Longitude", "Latitude", "Speed", "NetworkMode", "RSRP", "RSRQ", "DL_bitrate", "UL_bitrate", "State"])

#Printing top rows of car dataset
print(CarData.head())

#Turning the timestamp into more manipulable date and time columns
CarData[["Date", "Time"]] = CarData["Timestamp"].str.split("_", expand=True)

#Printing top rows of modified dataset
print(CarData.head())

#Declaring array for scenario markers
ScenarioMarkers = []

#Adding a relevant scenario marker to each record in the dataset
for i in range(0, len(CarData)):
    ScenarioMarkers.append("Car")

#Assigning a scenario marker to each record in the dataset
CarDataWithScenario = CarData.assign(Scenario=ScenarioMarkers)

#Printing top rows of final dataset
print(CarDataWithScenario.head())

#Writing final dataset to a CSV file
CarDataWithScenario.to_csv("FinalCarDataset.csv", index=False)
