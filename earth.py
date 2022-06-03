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
        "link for GUI code  [link](https://github.com/chirag81/"
        "Snapsolution/commit/b226eac802dff1f5338ea00a8b934b279481ecab#diff-"
        "7fb15ea9366150ab90b110a8dcdf79e3e5bd07054c165bdd503680074eca3b20)")
    st.write(
        "link for main code [link](https://github.com/chirag81/"
        "Snapsolution/commit/b226eac802dff1f5338ea00a8b934b279481ecab#diff"
        "-657107aa363a579581178acfa04f1ab6f1a68bd14e733bd805b05e79e2f1d47f)")
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
        st.write(pred)

