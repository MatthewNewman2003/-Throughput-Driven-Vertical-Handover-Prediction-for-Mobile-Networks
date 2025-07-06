#Importing libraries
import pandas as pd
import os

#Declaring folder path of bus dataset files (This will differ if executed on a different computer, apart from the final "bus" part)
FolderPath = r'C:\Users\Matthew Newman\Documents\University of Gloucestershire\BSc Hons Computer Science\Level 6\CT6039 Dissertation\Validation Dataset\Dataset\bus'

#Listing all files in the folder path
AllFiles = os.listdir(FolderPath)

#Filtering files in the folder path so that only CSV files are used
CSVFiles = [f for f in AllFiles if f.endswith('.csv')]

#Declaring list to store DataFrames
DataFrameList = []

#Reading each CSV within the folder path and adding it to the DataFrame list
#If errors are encountered, the reading is terminated and the user is informed
for csv in CSVFiles:
    FilePath = os.path.join(FolderPath, csv)
    try:
        DataFrame = pd.read_csv(FilePath)
        DataFrameList.append(DataFrame)
    except UnicodeDecodeError:
        try:
            DataFrame = pd.read_csv(FilePath, sep='\t', encoding='utf-16')
            DataFrameList.append(DataFrame)
        except Exception as e:
            print(f"Could not read the file {csv} because of error: {e}")
    except Exception as e:
        print(f"Could not read the file {csv} because of error: {e}")

#Combining all DataFrames
CombinedDataFrame = pd.concat(DataFrameList, ignore_index=True)

#Dropping index column from combined dataset
CombinedDataFrame.drop(["Unnamed: 0"], inplace=True, axis=1)

#Writing combined dataset to a CSV file
CombinedDataFrame.to_csv(os.path.join(FolderPath, "CombinedBusDataset.csv"), index=False)
            
