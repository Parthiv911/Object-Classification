import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.layers.normalization.batch_normalization import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt

import os
#Uncomment when using Windows if CuDNN does not work
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.datasets import cifar10

import tensorflow as tf

#Loading Data Set
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

#One Hot Encoding of Output Classes (The class 1 is represented by [1 0 0 0 0 0 0 0 0 0])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Normalizing the Data
x_test=x_test/255
x_train=x_train/255

#Constructing the Convolutional Neural Network
model=Sequential()
model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
model.add(BatchNormalization())

model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10,activation='softmax'))

#Compiling and training the network
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,verbose=1)

#Testing the network
score=model.evaluate(x_test,y_test,verbose=1)

#Printing Accuracy Score
print(score)