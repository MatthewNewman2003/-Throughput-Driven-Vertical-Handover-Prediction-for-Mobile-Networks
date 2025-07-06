#Importing Pandas
import pandas as pd

#Reading CSV
DataFrame = pd.read_csv("FinalCombinedDataset.csv")

#Finding LTE (4G) data and HSPA+ (3G) data within the dataset
FourGData = DataFrame[DataFrame["NetworkMode"] == "LTE"]
print(FourGData.head())
ThreeGData = DataFrame[DataFrame["NetworkMode"] == "HSPA+"]
print(ThreeGData.head())

#Cleaning the data so that only 4G and 3G data is kept and writing the cleaned dataset to a CSV file
Data = [FourGData, ThreeGData]
CleanedData = pd.concat(Data)
print(CleanedData.head())
CleanedData.to_csv("Filtered Combined Dataset (All).csv", index=False)
