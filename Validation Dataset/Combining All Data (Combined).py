#Importing Pandas
import pandas as pd

#Declaring list to store DataFrames
DataFrameList = []

#Reading CSVs for each mobility scenario individually
BusDataFrame = pd.read_csv("FinalBusDataset.csv")
CarDataFrame = pd.read_csv("FinalCarDataset.csv")
PedestrianDataFrame = pd.read_csv("FinalPedestrianDataset.csv")
StaticDataFrame = pd.read_csv("FinalStaticDataset.csv")
TrainDataFrame = pd.read_csv("FinalTrainDataset.csv")

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
CombinedDataFrame.to_csv("FinalCombinedDataset.csv", index=False)
