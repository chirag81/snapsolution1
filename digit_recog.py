import streamlit as st
from tensorflow.keras import models
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

import streamlit as st
def app():
    st.title("Prediction of digits")
    st.write("you can do classification of digits here")
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
    st.write(
        "link for GUI code  [link](https://github.com/chirag81/"
        "Snapsolution/commit/b226eac802dff1f5338ea00a8b934b279481ecab#diff"
        "-2289f55b0f32769ab0b95885432ed417a96c0853cf06a858b94f9ea4c4a1229b)")
    st.write(
        "link for main code [link](https://github.com/chirag81/"
        "Snapsolution/commit/b226eac802dff1f5338ea00a8b934b279481ecab#diff-"
        "891499e3eb56263f887dd593699090ec2e0514436f53efc84760a5a2a7f2f28c)")

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

        st.write(img_pred)

