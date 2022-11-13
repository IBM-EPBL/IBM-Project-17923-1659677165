
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

x_train=train_datagen.flow_from_directory(directory=r'D:\Python\train I test\data\train',target_size=(64,64),batch_size=32,class_mode='categorical')
x_test=test_datagen.flow_from_directory(directory=r'D:\Python\train I test\data\test',target_size=(64,64),batch_size=32,class_mode='categorical')

import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(units=128,kernel_initializer="random_uniform"))
model.add(Dense(6,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


model.fit(x_train,steps_per_epoch=len(x_train),epochs=10,validation_data=x_test,validation_steps=len(x_test))
model.save('ECG.h5')

