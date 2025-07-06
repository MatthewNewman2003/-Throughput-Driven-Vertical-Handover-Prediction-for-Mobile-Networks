#Importing Pandas
import pandas as pd

#Reading CSV and using the relevant columns
BusData = pd.read_csv("CombinedBusDataset.csv", usecols=["Timestamp", "Longitude", "Latitude", "Speed", "NetworkMode", "RSRP", "RSRQ", "DL_bitrate", "UL_bitrate", "State"])

#Printing top rows of bus dataset
print(BusData.head())

#Turning the timestamp into more manipulable date and time columns
BusData[["Date", "Time"]] = BusData["Timestamp"].str.split("_", expand=True)

#Printing top rows of the modified dataset
print(BusData.head())

#Creating array for scenario markers
ScenarioMarkers = []

#Adding a scenario marker for each record in the dataset
for i in range(0, len(BusData)):
    ScenarioMarkers.append("Bus")

#Assigning a scenario marker to each record in the dataset
BusDataWithScenario = BusData.assign(Scenario=ScenarioMarkers)

#Printing top rows of final dataset
print(BusDataWithScenario.head())

#Writing final dataset to a CSV file
BusDataWithScenario.to_csv("FinalBusDataset.csv", index=False)
