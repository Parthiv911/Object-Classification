import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.layers.normalization.batch_normalization import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt

import os
#Uncomment when using Windows if CuDNN does not work
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.datasets import cifar10

import tensorflow as tf

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_test=x_test/255
x_train=x_train/255

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=2,verbose=2)

score=model.evaluate(x_test,y_test,verbose=1)

print(score)