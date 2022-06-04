from tensorflow.keras import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


def app():
    st.title('Prediction of differnent places on earth')
    st.write("you can do clasification of different places of earth here")
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
    st.write(
        "link for GUI code  [link](https://github.com/chirag81/snapsolution1/blob/main/earth.py)")
    st.write(
        "link for main code [link](https://github.com/chirag81/snapsolution1/blob/main/earth1.py)")
    class_names = ['buildings',
                       'forest',
                       'glacier',
                       'mountain',
                       'sea',
                       'street']
    model = load_model('earth.h5')
    if image_file is not None:
        image = Image.open(image_file)
        img1 = image.resize((128, 128))
        img1 = preprocessing.image.img_to_array(img1)
        img1 = img1 / 255.0
        img1 = np.expand_dims(img1, axis=0)
        st.image(img1)
        #plt.imshow(image)
        #plt.axis('off')
        pred = class_names[np.argmax(model.predict(img1))]
        #plt.imshow(image)
        #plt.axis('off')
        st.write('The place shown in the image is:-',pred)

