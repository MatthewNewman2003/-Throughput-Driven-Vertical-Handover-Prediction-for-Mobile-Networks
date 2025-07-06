#Importing Pandas
import pandas as pd

#Declaring list to store DataFrames
DataFrameList = []

#Reading CSVs for each mobility scenario individually
BusDataFrame = pd.read_csv("FinalBusDataset (Status Quo).csv")
CarDataFrame = pd.read_csv("FinalCarDataset (Status Quo).csv")
PedestrianDataFrame = pd.read_csv("FinalPedestrianDataset (Status Quo).csv")
StaticDataFrame = pd.read_csv("FinalStaticDataset (Status Quo Variables).csv")
TrainDataFrame = pd.read_csv("FinalTrainDataset (Status Quo Variables).csv")

#Adding each mobility scenario's dataset to the DataFrame list
DataFrameList.append(BusDataFrame)
DataFrameList.append(CarDataFrame)
DataFrameList.append(PedestrianDataFrame)
DataFrameList.append(StaticDataFrame)
DataFrameList.append(TrainDataFrame)

#Combining all datasets
CombinedDataFrame = pd.concat(DataFrameList, ignore_index=True)

#Printing a summary of the final dataset
print(CombinedDataFrame)

#Writing the final dataset to a CSV file
CombinedDataFrame.to_csv("FinalCombinedDataset (Status Quo Variables).csv", index=False)
