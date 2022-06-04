import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model


def app():
    st.title("Prediction of digits")
    st.write("you can do classification of digits here")
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
    st.write(
        "link for GUI code  [link](https://github.com/chirag81/snapsolution1/blob/main/digit._recog.py)")
    st.write(
        "link for main code [link](https://github.com/chirag81/snapsolution1/blob/main/digit._recog1.py)")

    num1 =['0','1','2','3','4','5','6','7','8','9']
    model = load_model('mnist.h5')
    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR")

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 440))

        img = cv2.GaussianBlur(img, (7, 7), 0)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

        img_final = cv2.resize(img_thresh, (28, 28))
        img_final = np.reshape(img_final, (1, 28, 28, 1))

        img_pred = num1[np.argmax(model.predict(img_final))]

        st.write('The digit shown in the image is:-',img_pred)


