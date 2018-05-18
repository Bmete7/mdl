# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:54:47 2018

@author: BurakBey
"""





import numpy as np
import pandas as pd
import datetime
import time 
import math

dataset= pd.read_csv('train.txt')
labels = dataset.iloc[:,-1].values
features = dataset.iloc[:,:-1].values


def dateConversion(j):
    min_time = math.inf
    count = 0
    for i in features[:,j]: 
        if(i=='?'):
            features[count,j]=-1
        else:
            orderDate = datetime.datetime(int(i.split('-')[0]),int(i.split('-')[1]),int(i.split('-')[2]))
            
            time_num = orderDate.timestamp()
            time_num/=1000000
            features[count,j] = time_num
            min_time = min(min_time, time_num)            
        count+=1

    for i in range(len(features[:,j])):  
        if (features[i,j] != -1):
            features[i,j]  = float(("%0.2f"%(features[i,j] - min_time)))    

    dataset[1,j] = features[1,j]

dateConversion(1)
dateConversion(2)
#dateConversion(10)
dateConversion(12)



from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='Nan', strategy ='mean', axis = 0)
imputer = imputer.fit(features[:,1:2])
features[:,1] = imputer.transform(features[:,1])


