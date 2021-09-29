# -*- coding: utf-8 -*-
"""
@author: Megha
"""
#importing the libraries
import numpy as np #which allows us to work with arrays
import pandas as pd 
import matplotlib.pyplot as plt

#upload/import the dataset
dataset = pd.read_csv('Data.csv')
#any ML model will have featureset and dependent variable 
x = dataset.iloc[:, :-1].values  #belongs to featureset
y = dataset.iloc[:,-1].values   #considering the last col, dependent variable
#print(x)
print(y)

#Taking care of the missing data
#since we dont want to have any missing data in dataset, we need to handle the data
#we can replace the missing data by substituing the avegare of the data/median
#we can use scikit-learn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3]) #we will consider column1 and column2
x[:,1:3] = imputer.transform(x[:,1:3])
#print(x)

#Encoding the categorical data
#Encoding the independent variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#print(x)

#Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
#splitting the data to training and test set
#We have to apply feature scaling only after the training and test sets are split, so they are scaled equally or info leakage/data redundancey/ trian the data properly and test accuratley
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=1)

print("X_train",X_train)
print("X_test",X_test)
print("y_train",y_train)
print("y_test",y_test)

#Feature scaling
#Standardisation and Normalisation are the main feature scaling techniues
#where for standardisation the values will be stored between -3to+3, while in normalization the values will be stored in 0 and 1
#Normalization is recommanded when we have normal distribution
#Standardization is widely recommended, as it works for any kind of data.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#Note that applying standardization to dummy variables is not a good idea, 
#So its better to apply for the feature variables
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.transform(X_test[:, 3:])
print("X-train", X_train)
print("X_test", X_test)
