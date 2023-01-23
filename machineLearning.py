#!/usr/bin/env python
# coding: utf-8

# In[63]:


import preprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from  sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix, classification_report
from  sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

class MachineLearning:
    
    def __init__(self):
        ''' Applies a number of machine learning algorithms found in the sklearn library '''
        self.__dataframe = preprocess.Preprocessing().getDataset()
        # Establishes DEATH_EVENT as the target variable
        self.__X = self.__dataframe.drop('DEATH_EVENT', axis=1) # This may need to change for non-superived learning techniques
        self.__y = self.__dataframe['DEATH_EVENT']
        
        # splits data into training and test sets
        self.splitData()
        
    def setModel(self,model):
        ''' Sets the model type to be fitted '''
        self.model = model
    
    def getXTrain(self):
        ''' Returns the X_train array '''
        return self.__X_train
        
    def getYTrain(self):
        ''' Returns the y_train array '''
        return self.__y_train
        
    def getXTest(self):
        ''' Returns the X_test array '''
        return self.__X_test
    
    def getYTest(self):
        ''' Returns the y_test array '''
        return self.__y_test
    
    def splitData(self):
        ''' Splits data into training and test set '''
        # Split data
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X,self.__y,test_size=0.25)
        
        # Scale data
        scaler = StandardScaler()
        self.__X_train = scaler.fit_transform(self.__X_train)
        self.__X_test = scaler.fit_transform(self.__X_test)
        
    def knn(self):
        ''' Fits K-Nearest Neighbor model '''
        model = KNeighborsClassifier(n_neighbors=3)
        self.model = model.fit(self.__X_train,self.__y_train)
    
    def randomForest(self):
        ''' Fits Naive Bayes model '''
        model = RandomForestClassifier(n_estimators=150)
        self.model = model.fit(self.__X_train,self.__y_train)
    
    def logisticRegression(self):
        ''' Fits Logistic Regression model '''
        model = LogisticRegression()
        self.model = model.fit(self.__X_train,self.__y_train)

    def svm(self):
        ''' Fits Support Vector Machine model '''
        model = SVC(kernel='rbf', C=1, gamma='auto')
        self.model = model.fit(self.__X_train,self.__y_train)
    
    def naiveBayes(self):
        ''' Fits Naive Bayes model '''#
        model = GaussianNB()
        self.model = model.fit(self.__X_train,self.__y_train)
    
    def neuralNet(self):
        ''' Fits Neural Net model '''
        model = MLPClassifier(hidden_layer_sizes=(5,2), max_iter=10000, random_state=42, solver = 'adam')
        self.model = model.fit(self.__X_train,self.__y_train)
        
    def predict(self):
        ''' Predicts value given fitted model '''
        return self.model.predict(self.__X_test)
    
class EvaluateModel(MachineLearning):
    
    def __init__(self):
        ''' Provides evaluation metric for the models '''
        super().__init__()
    
    def confusionMatrix(self):
        ''' Returns a confusion matrix for the model '''
        return confusion_matrix(super().getYTest(),super().predict())
    
    def classificationReport(self):
        ''' Returns a classification report ''' 
        # Make more readable by converting to dataframe
        report = classification_report(super().getYTest(),super().predict(),output_dict=True)
        return pd.DataFrame(report)
'''
m = machineLearning()
m.neuralNet()
m.predict()
e = EvaluateModel()
e.randomForest()
e.confusionMatrix()
#e.classificationReport()
'''

# In[ ]:




