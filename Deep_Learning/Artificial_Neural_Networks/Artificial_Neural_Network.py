# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 10:53:09 2018

@author: bhanu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# encoding the variables 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
X[:, 1] = labelencoder_x_1.fit_transform(X[:, 1])
labelencoder_x_2 = LabelEncoder()
X[:, 2] = labelencoder_x_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:12]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# make the ANN


#importing the keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# initialising the ANN
classifier = Sequential()
# adding the input and the hiddden layers
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# adding the third hidden layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# compiling the neural network
classifier.compile(optimizer = 'adam',loss = "binary_crossentropy",metrics = ['accuracy'])

# fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size=10, epochs=100)

 
#making the prediction and evaluation of model




# predicting the test set results
y_pred = classifier.predict(X_test)
y_pred  = (y_pred>0.5)
l = [[0,0,470,1,32,7,52000,2,1,0,30000]]
l = sc_X.transform(l)
prediction = classifier.predict(l)
# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# accuracy matrix
from sklearn.metrics import accuracy_score
Score = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)