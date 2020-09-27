import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import keras
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.applications import ResNet50


df1 = pd.read_csv("../input/fer2013/fer2013.csv")

print(df1.emotion.value_counts())
print(df1.head())

# Preprocessing
x_train=[]
x_test=[]
y_train=[]
y_test=[]
for i,row in df1.iterrows():
    k=row['pixels'].split(" ")
    if(row['Usage']=='Training'):
        x_train.append(np.array(k))
        y_train.append(row['emotion'])
    elif(row['Usage']=='PublicTest'):
        x_test.append(np.array(k))
        y_test.append(row['emotion'])

x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

x_train=x_train.reshape(x_train.shape[0],48,48)
x_test=x_test.reshape(x_test.shape[0],48,48)
y_train=tf.keras.utils.to_categorical(y_train,num_classes=7)
y_test=tf.keras.utils.to_categorical(y_test,num_classes=7)

import matplotlib.pyplot as plt
for i in range(10):
  image=x_test[i].reshape((48,48))
  image=image.astype('float32')
  print(image.shape)
  plt.imshow(image,cmap=plt.cm.gray)
  plt.show()

#data augmentation
x_train=x_train.reshape((x_train.shape[0],48,48,1))
x_test=x_test.reshape((x_test.shape[0],48,48,1))
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=60,
                                   shear_range=0.5,
                                   zoom_range=0.5,
                                   width_shift_range=0.5,
                                   height_shift_range=0.5,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_datagen.fit(x_train)
validation_datagen.fit(x_test)

print(x_train.shape)


model1=keras.models.Sequential()

# Block-1
model1.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', 
                     kernel_initializer='he_normal',
                     activation="elu", 
                     input_shape=(48,48,1)))
model1.add(keras.layers.BatchNormalization())

model1.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', 
                     kernel_initializer='he_normal', 
                     activation="elu"))

model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model1.add(keras.layers.Dropout(0.2))

# Block-2
model1.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', 
                     kernel_initializer='he_normal',
                     activation="elu"))

model1.add(keras.layers.BatchNormalization())

model1.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',
                     kernel_initializer='he_normal', 
                     activation="elu"))

model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model1.add(keras.layers.Dropout(0.2))

# Block-3
model1.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', 
                     kernel_initializer='he_normal', 
                     activation="elu"))
model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', 
                     kernel_initializer='he_normal',
                     activation="elu"))

model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model1.add(keras.layers.Dropout(0.2))

# Block-4
model1.add(keras.layers.Flatten())
model1.add(keras.layers.Dense(64, activation="elu", kernel_initializer='he_normal'))
model1.add(keras.layers.BatchNormalization())
model1.add(keras.layers.Dropout(0.5))

# Block-5
model1.add(keras.layers.Dense(7, activation="softmax", kernel_initializer='he_normal'))

print(model1.summary())

#Model Plot
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils

keras.utils.plot_model(model1, to_file='model.png', show_layer_names=True)

x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

#intializing callbacks
early_stopping=keras.callbacks.EarlyStopping(patience=15,restore_best_weights=True)
filepath="weights/weights.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model1.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


model1.fit(x_train,y_train,
           batch_size=64,
           epochs=50,
           validation_data=(x_test,y_test),
           verbose=1,callbacks=[early_stopping])

print(model1.evaluate(x_test,y_test))

fer_json = model1.to_json()  
with open("fer.json", "w") as json_file:  
    json_file.write(fer_json)  
model1.save_weights("fer.h5")

