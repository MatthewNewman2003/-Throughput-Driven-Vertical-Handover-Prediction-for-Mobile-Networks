#Importing Pandas
import pandas as pd

#Reading CSV, with only the desired columns selected
DataFrame = pd.read_csv("Throughput Tests - Speedtest - Active Measurements.csv", usecols=["Date", "Time", "GPS Long", "GPS Lat", "RAT Info", "Current Netw. DL", "Current Netw. UL", "Mean Netw. DL", "Mean Netw. UL", "Current Netw. DL Avg", "Current Netw. UL Avg", "Current Netw. DL Max", "Current Netw. UL Max", "UE Mode", "Scenario"])

#Printing summary of DataFrame
print(DataFrame)

#Filtering data so that only records with 5G enabled were used
UEModeFilteredData = DataFrame.loc[DataFrame["UE Mode"] == "5G-enabled"]

#Printing summary of filtered data
print(UEModeFilteredData)

#Writing filtered data to a new CSV file
UEModeFilteredData.to_csv("Filtered Throughput Dataset.csv")
