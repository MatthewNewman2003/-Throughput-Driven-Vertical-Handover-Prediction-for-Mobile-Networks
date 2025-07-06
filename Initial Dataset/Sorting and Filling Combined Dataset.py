#Importing Pandas
import pandas as pd

#Declaring missing value denoter
MissingValues = ["?"]

#Reading CSV
DataFrame = pd.read_csv("Filtered Combined Dataset.csv", na_values=MissingValues)

#Dropping index column
DataFrame = DataFrame.drop("Unnamed: 0", axis=1)

#Filtering the dataset to only show data for Date 1
Date1FilteredData = DataFrame.loc[DataFrame["Date"] == "04.01.2021"]

#Printing a summary of the Date 1 dataset
print(Date1FilteredData)

#Sorting the Date 1 dataset by time
Date1SortedData = Date1FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date1SortedData)

#Forward-filling the Date 1 dataset to fill in any gaps
Date1FilledData = Date1SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date1FilledData)

#Writing final Date 1 dataset to a CSV file
Date1FilledData.to_csv("Date 1 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 2
Date2FilteredData = DataFrame.loc[DataFrame["Date"] == "05.01.2021"]

#Printing a summary of the Date 2 dataset
print(Date2FilteredData)

#Sorting the Date 2 dataset by time
Date2SortedData = Date2FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date2SortedData)

#Forward-filling the Date 2 dataset to fill in any gaps
Date2FilledData = Date2SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date2FilledData)

#Writing final Date 2 dataset to a CSV file
Date2FilledData.to_csv("Date 2 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 3
Date3FilteredData = DataFrame.loc[DataFrame["Date"] == "13.12.2020"]

#Printing a summary of the Date 3 dataset
print(Date3FilteredData)

#Sorting the Date 3 dataset by time
Date3SortedData = Date3FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date3SortedData)

#Forward-filling the Date 3 dataset to fill in any gaps
Date3FilledData = Date3SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date3FilledData)

#Writing final Date 3 dataset to a CSV file
Date3FilledData.to_csv("Date 3 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 4
Date4FilteredData = DataFrame.loc[DataFrame["Date"] == "27.01.2021"]

#Printing a summary of the Date 4 dataset
print(Date4FilteredData)

#Sorting the Date 4 dataset by time
Date4SortedData = Date4FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date4SortedData)

#Forward-filling the Date 4 dataset to fill in any gaps
Date4FilledData = Date4SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date4FilledData)

#Writing final Date 4 dataset to a CSV file
Date4FilledData.to_csv("Date 4 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 5
Date5FilteredData = DataFrame.loc[DataFrame["Date"] == "07.01.2021"]

#Printing a summary of the Date 5 dataset
print(Date5FilteredData)

#Sorting the Date 5 dataset by time
Date5SortedData = Date5FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date5SortedData)

#Forward-filling the Date 5 dataset to fill in any gaps
Date5FilledData = Date5SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date5FilledData)

#Writing final Date 5 dataset to a CSV file
Date5FilledData.to_csv("Date 5 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 6
Date6FilteredData = DataFrame.loc[DataFrame["Date"] == "15.12.2020"]

#Printing a summary of the Date 6 dataset
print(Date6FilteredData)

#Sorting the Date 6 dataset by time
Date6SortedData = Date6FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date6SortedData)

#Forward-filling the Date 6 dataset to fill in any gaps
Date6FilledData = Date6SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date6FilledData)

#Writing final Date 6 dataset to a CSV file
Date6FilledData.to_csv("Date 6 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 7
Date7FilteredData = DataFrame.loc[DataFrame["Date"] == "24.01.2021"]

#Printing a summary of the Date 7 dataset
print(Date7FilteredData)

#Sorting the Date 7 dataset by time
Date7SortedData = Date7FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date7SortedData)

#Forward-filling the Date 7 dataset to fill in any gaps
Date7FilledData = Date7SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date7FilledData)

#Writing final Date 7 dataset to a CSV file
Date7FilledData.to_csv("Date 7 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 8
Date8FilteredData = DataFrame.loc[DataFrame["Date"] == "08.01.2021"]

#Printing a summary of the Date 8 dataset
print(Date8FilteredData)

#Sorting the Date 8 dataset by time
Date8SortedData = Date8FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date8SortedData)

#Forward-filling the Date 8 dataset to fill in any gaps
Date8FilledData = Date8SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date8FilledData)

#Writing final Date 8 dataset to a CSV file
Date8FilledData.to_csv("Date 8 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 9
Date9FilteredData = DataFrame.loc[DataFrame["Date"] == "16.12.2020"]

#Printing a summary of the Date 9 dataset
print(Date9FilteredData)

#Sorting the Date 9 dataset by time
Date9SortedData = Date9FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date9SortedData)

#Forward-filling the Date 9 dataset to fill in any gaps
Date9FilledData = Date9SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date9FilledData)

#Writing final Date 9 dataset to a CSV file
Date9FilledData.to_csv("Date 9 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 10
Date10FilteredData = DataFrame.loc[DataFrame["Date"] == "26.01.2021"]

#Printing a summary of the Date 10 dataset
print(Date10FilteredData)

#Sorting the Date 10 dataset by time
Date10SortedData = Date10FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date10SortedData)

#Forward-filling the Date 10 dataset to fill in any gaps
Date10FilledData = Date10SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date10FilledData)

#Writing final Date 10 dataset to a CSV file
Date10FilledData.to_csv("Date 10 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 11
Date11FilteredData = DataFrame.loc[DataFrame["Date"] == "14.01.2021"]

#Printing a summary of the Date 11 dataset
print(Date11FilteredData)

#Sorting the Date 11 dataset by time
Date11SortedData = Date11FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date11SortedData)

#Forward-filling the Date 11 dataset to fill in any gaps
Date11FilledData = Date11SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date11FilledData)

#Writing final Date 11 dataset to a CSV file
Date11FilledData.to_csv("Date 11 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 12
Date12FilteredData = DataFrame.loc[DataFrame["Date"] == "06.01.2021"]

#Printing a summary of the Date 12 dataset
print(Date12FilteredData)

#Sorting the Date 12 dataset by time
Date12SortedData = Date12FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date12SortedData)

#Forward-filling the Date 12 dataset to fill in any gaps
Date12FilledData = Date12SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date12FilledData)

#Writing final Date 12 dataset to a CSV file
Date12FilledData.to_csv("Date 12 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 13
Date13FilteredData = DataFrame.loc[DataFrame["Date"] == "17.12.2020"]

#Printing a summary of the Date 13 dataset
print(Date13FilteredData)

#Sorting the Date 13 dataset by time
Date13SortedData = Date13FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date13SortedData)

#Forward-filling the Date 13 dataset to fill in any gaps
Date13FilledData = Date13SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date13FilledData)

#Writing final Date 13 dataset to a CSV file
Date13FilledData.to_csv("Date 13 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 14
Date14FilteredData = DataFrame.loc[DataFrame["Date"] == "25.01.2021"]

#Printing a summary of the Date 14 dataset
print(Date14FilteredData)

#Sorting the Date 14 dataset by time
Date14SortedData = Date14FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date14SortedData)

#Forward-filling the Date 14 dataset to fill in any gaps
Date14FilledData = Date14SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date14FilledData)

#Writing final Date 14 dataset to a CSV file
Date14FilledData.to_csv("Date 14 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 15
Date15FilteredData = DataFrame.loc[DataFrame["Date"] == "14.12.2020"]

#Printing a summary of the Date 15 dataset
print(Date15FilteredData)

#Sorting the Date 15 dataset by time
Date15SortedData = Date15FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date15SortedData)

#Forward-filling the Date 15 dataset to fill in any gaps
Date15FilledData = Date15SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date15FilledData)

#Writing final Date 15 dataset to a CSV file
Date15FilledData.to_csv("Date 15 Cleaned Combined Data.csv")

#Filtering the dataset to only show data for Date 16
Date16FilteredData = DataFrame.loc[DataFrame["Date"] == "11.01.2021"]

#Printing a summary of the Date 16 dataset
print(Date16FilteredData)

#Sorting the Date 16 dataset by time
Date16SortedData = Date16FilteredData.sort_values(by="Time", ascending=True)

#Printing a summary of the sorted dataset
print(Date16SortedData)

#Forward-filling the Date 16 dataset to fill in any gaps
Date16FilledData = Date16SortedData.ffill(axis = 0)

#Printing a summary of the filled dataset
print(Date16FilledData)

#Writing final Date 16 dataset to a CSV file
Date16FilledData.to_csv("Date 16 Cleaned Combined Data.csv")

#Concatenating all datasets together
CombinedDataset = pd.concat([Date1FilledData, Date2FilledData, Date3FilledData, Date4FilledData, Date5FilledData, Date6FilledData, Date7FilledData, Date8FilledData, Date9FilledData, Date10FilledData, Date11FilledData, Date12FilledData, Date13FilledData, Date14FilledData, Date15FilledData, Date16FilledData])

#Writing final combined dataset to a CSV file
CombinedDataset.to_csv("Final Combined Dataset.csv")
