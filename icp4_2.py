# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:06:38 2019

@author: ahmed
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB #to build classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

#Reading the dataset 'glass'

data_set = pd.read_csv("glass.csv")

#Preprocessing

x = data_set.drop('Type',axis=1)
y = data_set['Type']

#Train Test Split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=30)

#Training the classifier

model = GaussianNB()
model.fit(x_train, y_train)

#Making Predictions

y_pred = model.predict(x_test)

#Evaluating the model
print("Classification report is: \n", classification_report(y_test, y_pred))
print("Accuracy score is: ", accuracy_score(y_test, y_pred))