# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:01:03 2019

@author: ahmed
"""

import pandas as pd
dataset=pd.read_csv('train.csv')
dataset['Sex']=dataset['Sex'].map({'female':1,'male':0})
print("correlation score  for sex is",dataset['Survived'].corr(dataset['Sex']))