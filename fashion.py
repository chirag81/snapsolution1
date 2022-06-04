import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
def app():
    st.title('Prediction of clothes')
    st.write("you can do clasification of different clothes here")
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
    st.write(
        "link for GUI code  [link](https://github.com/chirag81/snapsolution1/blob/main/fashion.py)")
    st.write(
        "link for main code [link](https://github.com/chirag81/snapsolution1/blob/main/fashion1.py)")

    class_n = {0: 'T_shirt/top',
               1: 'Trouser',
               2: 'Pullover',
               3: 'Dress',
               4: 'Coat',
               5: 'Sandals',
               6: 'Shirt',
               7: 'Sneaker',
               8: 'Bag',
               9: 'Ankle boot'}

    model = load_model('fashion1.h5')
    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR")
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_copy = cv2.GaussianBlur(img, (7, 7), 0)
        img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

        img_final = cv2.resize(img_thresh, (28, 28))
        img_final = np.reshape(img_final, (1, 28, 28, 1))
        st.image(img_final)

        # img = image.load_img(img1, target_size=(30, 30))
        #img = np.array(img)
        #img = np.expand_dims(img, axis=0)
        #img = preprocess_input(img)
        # pred = model.predict(img4)
        pred = class_n[np.argmax(model.predict(img_final))]
        # pred = classes1[model.predict(img4)]
        # pred = model.predict_classes([image])[0]
        st.write(pred)




