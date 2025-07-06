#Importing Pandas
import pandas as pd

#Reading CSV and using the relevant columns
BusData = pd.read_csv("CombinedBusDataset.csv", usecols=["Timestamp", "Longitude", "Latitude", "Speed", "NetworkMode", "RSRP", "RSRQ", "State"])

#Printing top rows of bus dataset
print(BusData.head())

#Turning the timestamp into more manipulable date and time columns
BusData[["Date", "Time"]] = BusData["Timestamp"].str.split("_", expand=True)

#Printing top rows of the modified bus dataset
print(BusData.head())

#Declaring array for scenario markers
ScenarioMarkers = []

#Adding a scenario marker of "Bus" for each record in the bus dataset
for i in range(0, len(BusData)):
    ScenarioMarkers.append("Bus")

#Assigning a scenario marker to each record in the dataset
BusDataWithScenario = BusData.assign(Scenario=ScenarioMarkers)

#Printing top rows of final bus dataset
print(BusDataWithScenario.head())

#Writing final dataset to a CSV file
BusDataWithScenario.to_csv("FinalBusDataset (Status Quo).csv", index=False)
