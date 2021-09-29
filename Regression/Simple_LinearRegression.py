# -*- coding: utf-8 -*-
"""
@author: Megha
"""
#Implementation of the Simple Linear Regression: which have only one feature vector.
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values  #belongs to featureset
y = dataset.iloc[:,-1].values   #considering the last col, dependent variable
#print(x)
#print(y)
#Split the dataset to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size=0.8, random_state = 1)

#Train on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict on the test set
y_pred = regressor.predict(X_test)

#Visualize the training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train,regressor.predict(X_train), color="blue")
plt.title('Salary Vs Experience - Training Set')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
#Visualize the test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train,regressor.predict(X_train), color="blue")
plt.title('Salary Vs Experience - Test Set')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#To know the value of a particular salary prediction of an employee with particular years(15 years) of experience, 
print(regressor.predict([[15]]))
#To predict using the Simple LinearRegression equation: y=b0+xb1
print(regressor.coef_)
print(regressor.intercept_)
print("Equation to calculate the salary:",9332.944*15+25609.89)
