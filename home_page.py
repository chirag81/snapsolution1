import streamlit as st
from PIL import Image
def app():
    st.title('snap solutions')

    st.subheader('welcome to the world of images')
    if image_file is not None:
        image = Image.open("C:\python help\cnn1.png")
        img1 = image.resize((128, 128))
        img1 = preprocessing.image.img_to_array(img1)
        img1 = np.expand_dims(img1, axis=0)
        st.image(img1)