# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 20:55:08 2017

@author: bhanu
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# fitting the training data in the simple linear regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predicting the y set

y_pred = regressor.predict(X_test)

# plot the sets so that we can see this pictorally this is training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('salary vs experience (training set)')
plt.xlabel('Years of Experiene')
plt.ylabel('Salary')
plt.show()# this show will tell the plotter to plot untill here from the start of the scatter

# this is the test set 
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')#we dont need to change this because this is the actual line of reference if we change we will get another one
plt.title('salary vs experience (test set)')
plt.xlabel('Years of Experiene')
plt.ylabel('Salary')
plt.show()
