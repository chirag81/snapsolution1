#traffic main code
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

imgs_path = r"C:\python help\traffic\Train"
data = []
labels = []
#classes = 43
for i in range(43):
    img_path = os.path.join(imgs_path,str(i))
    for img in os.listdir(img_path):
        im = Image.open(img_path + '/'+ img)
        im = im.resize((30,30))
        im = np.array(im)
        data.append(im)
        labels.append(i)

#data means x
data = np.array(data)
#y data
labels = np.array(labels)

x_train, x_test,y_train,y_test = train_test_split(data,labels,test_size=0.2,random_state=42)
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test,43)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape=x_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
epochs = 30
history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_data=(x_test, y_test))

test = pd.read_csv(r"C:\python help\traffic\Test.csv")
test_labels = test['ClassId'].values
test_imgs_path = r"C:\python help\traffic"
test_imgs = test['Path'].values
test_data = []
#test_lables = []
for img in test_imgs:
    im = Image.open(test_imgs_path + '/'+img)
    im = im.resize((30,30))
    im = np.array(im)
    test_data.append(im)

test_data = np.array(test_data)
predictions = model.predict(test_data)
#print(("accuracy:", accuracy_score(test_labels,predictions)))
#print(test_data.shape)
print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])
model.summary()
model.save('traffic_classifier.h5')