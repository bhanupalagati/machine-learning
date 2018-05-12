# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 13:26:14 2018

@author: bhanu
"""

# importing the packages 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#initialising thr CNN
classifier = Sequential()

# convolution part
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))


#Max Polling
classifier.add(MaxPooling2D(pool_size = (2,2)))


# adding the second convolution layer and polling layer
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flattening part
classifier.add(Flatten())


# Full Connection
# first hidden layer

classifier.add(Dense(output_dim = 128, activation="relu"))
# the output layer

classifier.add(Dense(output_dim = 1, activation="sigmoid"))

# compiling the classifier
classifier.compile(optimizer="adam", loss = "binary_crossentropy",metrics = ['accuracy'])


 #=============================================================================
# # fitting the classifier
from keras.preprocessing.image import ImageDataGenerator
# 
train_datagen = ImageDataGenerator( rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
 
test_datagen = ImageDataGenerator(rescale=1./255)
# 
training_data = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(64, 64),                                                     batch_size=32,
                                                     class_mode='binary')
#         
# 
test_data = test_datagen.flow_from_directory('dataset/test_set',
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='binary')
 
 #classifier.fit_generator(training_data,
  #                        steps_per_epoch=8000,
   #                       nb_epoch=25,
    #                      validation_data=test_data,
    #                      nb_val_samples=2000)
classifier.fit_generator(training_data,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_data,
                         nb_val_samples = 2000)



# real time 
from skimage.io import imread
from skimage.transform import resize
import numpy as np
img = imread("final_set/CAT.jpg") #make sure that path_to_file contains the path to the image you want to predict on. 
img = resize(img,(64,64))
img = np.expand_dims(img,axis=0)
img = img/(255.0)
prediction = classifier.predict_classes(img)
 
if(prediction):
    print ("CAT")
else:
    print ("DOG")
    
# =============================================================================



# =============================================================================
# from keras.preprocessing.image import ImageDataGenerator
# 
# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    horizontal_flip = True)
# 
# test_datagen = ImageDataGenerator(rescale = 1./255)
# 
# training_set = train_datagen.flow_from_directory('dataset/training_set',
#                                                  target_size = (64, 64),
#                                                  batch_size = 32,
#                                                  class_mode = 'binary')
# 
# test_set = test_datagen.flow_from_directory('dataset/test_set',
#                                             target_size = (64, 64),
#                                             batch_size = 32,
#                                             class_mode = 'binary')
# 
# classifier.fit_generator(training_set,
#                          samples_per_epoch = 8000,
#                          nb_epoch = 25,
#                          validation_data = test_set,
#                          nb_val_samples = 2000)
# 
# =============================================================================
