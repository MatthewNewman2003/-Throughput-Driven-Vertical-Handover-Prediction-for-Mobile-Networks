#Importing Pandas
import pandas as pd

#Reading CSV and using relevant columns
PedestrianData = pd.read_csv("CombinedPedestrianDataset.csv", usecols=["Timestamp", "Longitude", "Latitude", "Speed", "NetworkMode", "DL_bitrate", "UL_bitrate", "State"])

#Printing top rows of pedestrian dataset
print(PedestrianData.head())

#Turning the timestamp into more manipulable date and time columns
PedestrianData[["Date", "Time"]] = PedestrianData["Timestamp"].str.split("_", expand=True)

#Printing top rows of modified dataset
print(PedestrianData.head())

#Declaring array for scenario markers
ScenarioMarkers = []

#Adding a relevant scenario marker to each record in the dataset
for i in range(0, len(PedestrianData)):
    ScenarioMarkers.append("Pedestrian")

#Assigning a scenario marker to each record in the dataset
PedestrianDataWithScenario = PedestrianData.assign(Scenario=ScenarioMarkers)

#Printing top rows of final dataset
print(PedestrianDataWithScenario.head())

#Writing final dataset to a CSV file
PedestrianDataWithScenario.to_csv("FinalPedestrianDataset (Throughput).csv", index=False)
