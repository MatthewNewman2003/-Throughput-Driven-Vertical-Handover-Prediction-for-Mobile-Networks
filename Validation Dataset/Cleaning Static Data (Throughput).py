#Importing Pandas
import pandas as pd

#Reading CSV and using relevant columns
StaticData = pd.read_csv("CombinedStaticDataset.csv", usecols=["Timestamp", "Longitude", "Latitude", "Speed", "NetworkMode", "DL_bitrate", "UL_bitrate", "State"])

#Printing top rows of static dataset
print(StaticData.head())

#Turning the timestamp into more manipulable date and time columns
StaticData[["Date", "Time"]] = StaticData["Timestamp"].str.split("_", expand=True)

#Printing top rows of modified dataset
print(StaticData.head())

#Declaring array for scenario markers
ScenarioMarkers = []

#Adding a relevant scenario marker to each record in the dataset
for i in range(0, len(StaticData)):
    ScenarioMarkers.append("Static")

#Assigning a scenario marker to each record in the dataset
StaticDataWithScenario = StaticData.assign(Scenario=ScenarioMarkers)

#Printing top rows of final dataset
print(StaticDataWithScenario.head())

#Writing final dataset to a CSV file
StaticDataWithScenario.to_csv("FinalStaticDataset (Throughput).csv", index=False)
