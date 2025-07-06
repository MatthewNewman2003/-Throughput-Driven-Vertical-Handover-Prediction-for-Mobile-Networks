#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt

#Reading CSV
DataFrame = pd.read_csv("Filtered Combined Dataset (All).csv", na_values="-")

#Grouping dataset by the RAT connected to
GroupedDataFrame = DataFrame.groupby(["NetworkMode"]).count()

#Printing a summary of the grouped dataset
print(GroupedDataFrame)

#Creating a pie chart showing the proportion of each RAT
plt.pie(GroupedDataFrame["Longitude"], autopct='%1.0f%%')

#Showing pie chart
plt.show()

#Dropping any records with missing values from the dataset
CleanedDataFrame = DataFrame.dropna()

#Grouping dataset by the RAT connected to
GroupedCleanDataFrame = CleanedDataFrame.groupby(["NetworkMode"]).count()

#Printing a summary of the grouped dataset
print(GroupedCleanDataFrame)

#Creating a pie chart showing the proportion of each RAT
plt.pie(GroupedCleanDataFrame["Longitude"], autopct='%1.0f%%')

#Showing pie chart
plt.show()
