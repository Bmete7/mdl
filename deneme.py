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

# Manually encoding the dates, converting to floats with respect to its 
# time value(m seconds)
def dateConversion(j):
    min_time = math.inf
    count = 0
    global features
    for i in features[:,j]: 
        if(i=='?'):
            features[count,j]=-1
        else:
            if(int(i.split('-')[0]) < 1900):
                features = np.delete(features,count,0)
                count -= 1
            else:
                orderDate = datetime.datetime(int(i.split('-')[0])  + 100,int(i.split('-')[1]),int(i.split('-')[2]))            
                time_num = orderDate.timestamp()
                time_num/=1000000
                features[count,j] = time_num
                min_time = min(min_time, time_num)            
        count+=1

def missingDataDelete():
    global features
    count = 0
    for i in features[:,5]:
        if i == '?':
            features = np.delete(features,count,0)
            count -= 1
        count+= 1
missingDataDelete()

dateConversion(1)
dateConversion(2)
dateConversion(10)
dateConversion(12)
print('Date values cleaned')
editedData = pd.DataFrame(features) 

#Handle Missing data, make each missing instance of coloumn 2 its average

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=-1, strategy ='mean', axis = 0)

fitted_cols = features[:,1:3].reshape(len(features[:]),2)
imputer = imputer.fit(fitted_cols)
fitted_cols = imputer.transform(fitted_cols)
features[:,1:3] = fitted_cols
editedData = pd.DataFrame(features) 

fitted_cols = features[:,10].reshape(len(features[:]),1)
imputer = Imputer(missing_values=-1, strategy ='mean', axis = 0)

imputer = imputer.fit(fitted_cols)
fitted_cols = imputer.transform(fitted_cols)
features[:,10] = fitted_cols.reshape(len(features[:], ))
editedData = pd.DataFrame(features) 

print('Missing data is handled')


# Encoding for string values
for i in range(len(features[:])):
    features[i,4] =features[i,4].lower()
    features[i,9] =features[i,9].lower()
    features[i,11] =features[i,11].lower()

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelenc = LabelEncoder()

fitted_cols = features[:,4].reshape(len(features[:] ),1)
fitted_cols = labelenc.fit_transform(fitted_cols.ravel())
features[:,4] = fitted_cols.reshape(len(features[:]))
editedData = pd.DataFrame(features) 

fitted_cols = features[:,5].reshape(len(features[:] ),1)
fitted_cols = labelenc.fit_transform(fitted_cols.ravel())
features[:,5] = fitted_cols.reshape(len(features[:]))
editedData = pd.DataFrame(features) 

fitted_cols = features[:,9].reshape(len(features[:] ),1)
fitted_cols = labelenc.fit_transform(fitted_cols.ravel())
features[:,9] = fitted_cols.reshape(len(features[:]))
editedData = pd.DataFrame(features) 

fitted_cols = features[:,11].reshape(len(features[:] ),1)
fitted_cols = labelenc.fit_transform(fitted_cols.ravel())
features[:,11] = fitted_cols.reshape(len(features[:] ))
editedData = pd.DataFrame(features) 

print('String data encoded to int')