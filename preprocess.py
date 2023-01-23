#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')

class Preprocessing:

    def __init__(self,dataset='heart_failure_clinical_records_dataset.csv'):
        try:
            self.dataset = pd.read_csv(dataset)
        except:
            return 'File Not Found'

    def setDataset(self,dataset):
        ''' set Dataset to be inputted (must be in working directory) '''
        self.dataset = dataset

    def getDataset(self):
        ''' Inputs dataset (CSV) into a panda dataframe '''
        try:
            df = pd.read_csv(self.dataset)
            return df
        except: # If called without establishing a new csv file to read from
            return self.dataset

    def cleanDataset(self,dataframe):
        ''' Removes any rows with missing data '''
        df = dataframe.dropna()
        self.dataset = df

class StatAnalysis(Preprocessing):

    def __init__(self):
        super().__init__() 

    def getMean(self,attribute):
        ''' Returns mean of data in a column '''
        try:
            return self.dataset[attribute].mean()
        except:
            return "attribute not found"

    def getSTD(self,attribute):
        ''' Returns standard deviation of data '''
        try:
            return self.dataset[attribute].std()
        except:
            return "attribute not found"
        
    def getMedian(self,attribute):
        ''' Returns the median of data '''
        try:
            return self.dataset[attribute].median()
        except:
            return "attribute not found"

    def getVariance(self,attribute):
        ''' Returns the variance of data '''
        try:
            return self.dataset[attribute].variance()
        except:
            return "attribute not found" 

    def getMin(self,attribute):
        ''' Returns the minimum of data '''
        try:
            return self.dataset[attribute].min()
        except:
            return "attribute not found"
        
    def getMax(self,attribute):
        ''' Returns the maximum of data '''
        try:
            return self.dataset[attribute].max()
        except:
            return "attribute not found"

    def getSkewness(self,attribute):
        ''' Returns the skewness of data '''
        try:
            return self.dataset[attribute].skew()
        except:
            return "attribute not found" 

    def getKurtosis(self,attribute):
        ''' Returns the kurtosis of data '''
        try:
            return self.dataset[attribute].kurtosis()
        except:
            return "attribute not found"

class VisualizeData(Preprocessing):

    def __init__(self):
        super().__init__()
        
    def getBarPlot(self,attribute):
        ''' Returns bar plot '''
        
        unique_values, freq = np.unique(self.dataset[attribute], return_counts=True)
        
        # Plot chart
        plt.bar(unique_values, freq)
        plt.xlabel('%s'%attribute)
        plt.ylabel("Frequency")
        plt.show()
        
    def getGroupedBarPlot(self,attribute1,attribute2):
        ''' Returns the grouped bar plot of two columns in dataframe '''
        
        # Find the unique values and their frequencies for column1
        values1, frequencies1 = np.unique(self.dataset[attribute1], return_counts=True)

        # Find the unique values and their frequencies for column2
        values2, frequencies2 = np.unique(self.dataset[attribute2], return_counts=True)
        
        # Offset charts
        bar_width = 0.35
        
        # Plot charts
        plt.bar(values1, frequencies1, bar_width, label='%s'%attribute1)
        plt.bar(values2 + bar_width, frequencies2, bar_width, label='%s'%attribute2)
        
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    def getPieChart(self):
        ''' Returns a pie chart of the data '''
        pass
        
'''
test = Preprocessing()
#test.getDataset()
test2 = StatAnalysis()
#test2.getDataset()
ds = test2.getDataset()
ds = test2.getMean("age")
ds = test2.getKurtosis("age")

test3 = visualizeData()
test3.getBarPlot("DEATH_EVENT")
#test3.getGroupedBarPlot("diabetes","anaemia")
'''

# In[ ]:




