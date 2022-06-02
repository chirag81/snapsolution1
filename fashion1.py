#fashion code main
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
train = pd.read_csv(r"C:\python help\fashion-mnist\fashion-mnist_train.csv")
test = pd.read_csv(r"C:\python help\fashion-mnist\fashion-mnist_test.csv")

train1 = np.array(train,dtype='float32')
test1 = np.array(test,dtype='float32')

x_train = train1[:,1:]/255
y_train = train1[:,0]
x_test = test1[:,1:]/255
y_test = test1[:,0]

x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size=0.2,random_state=12345)
"""
x_train=np.array(x_train,dtype='float32')
x_validate=np.array(x_validate,dtype='float32')
y_train=np.array(y_train,dtype='float32')
y_validate=np.array(y_validate,dtype='float32')
x_test=np.array(x_test,dtype='float32')
"""

class_n =  ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""
plt.figure(figsize=(10,10))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i].reshape((28, 28)))
    label_index = int(y_train[i])
    plt.title(class_n[label_index])
plt.show()
"""
image_rows = 28
image_cols = 28
batch_size = 4096
image_shape = (image_rows,image_cols,1)

x_train =x_train.reshape(x_train.shape[0],*image_shape)
x_test=x_test.reshape(x_test.shape[0],*image_shape)
x_validate=x_validate.reshape(x_validate.shape[0],*image_shape)


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=image_shape))
model.add(MaxPool2D(pool_size=(2, 2), strides=1))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=1))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=1))
model.add(Flatten())

model.add(Dense(32,activation ="relu"))
model.add(Dense(128,activation ="relu"))

model.add(Dense(10,activation ="softmax"))



model.compile(optimizer = Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


history = model.fit(x_train, y_train, epochs=15, callbacks=[reduce_lr, early_stop],  validation_data = (x_validate,y_validate))

model.summary()
model.save(r'fashion1.h5')

#history = model.fit(x_train, y_train, epochs=75, callbacks=[reduce_lr, early_stop],  validation_data=(x_validate,y_validate))


#history = model.fit(x_train,y_train,batch_size=4096, epochs=75,verbose=1,validation_data=(x_validate,y_validate))


