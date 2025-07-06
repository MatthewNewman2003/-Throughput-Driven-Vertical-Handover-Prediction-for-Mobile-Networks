#Importing Pandas
import pandas as pd

#Reading CSV, with only the desired columns selected
DataFrame = pd.read_csv("Throughput Tests - Speedtest - Active Measurements.csv", usecols=["Date", "Time", "GPS Long", "GPS Lat", "RAT Info", "SS-RSRP", "SS-RSRQ", "SS-SINR", "UE Mode", "Scenario"])

#Printing summary of unfiltered data
print(DataFrame)

#Filtering data to only contain records where 5G was enabled
UEModeFilteredData = DataFrame.loc[DataFrame["UE Mode"] == "5G-enabled"]

#Printing summary of filtered data
print(UEModeFilteredData)

#Writing filtered data to a new CSV
UEModeFilteredData.to_csv("Filtered Status Quo Throughput Dataset.csv")
