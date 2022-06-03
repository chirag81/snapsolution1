from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

rescaled = ImageDataGenerator(1/255)
train_x = rescaled.flow_from_directory(r"C:\python help\world\seg_train\seg_train",target_size=(128,128),batch_size=32,class_mode='categorical')
test_x = rescaled.flow_from_directory(r"C:\python help\world\seg_test\seg_test",target_size=(128,128),batch_size=32,class_mode='categorical')


model = Sequential()


model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',input_shape=(128,128,3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(128,activation ="relu"))
model.add(Dropout(0.5))
model.add(Dense(6,activation ="softmax"))
model.summary()

model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


history = model.fit(train_x, epochs=20, callbacks=[reduce_lr, early_stop],  validation_data = (test_x))


model.summary()
model.save(r'earth.h5')



