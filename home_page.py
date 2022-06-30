import streamlit as st
from PIL import Image
def app():
    st.title('snap solutions')

    st.subheader('welcome to the world of images')
    image1 = Image.open(r"C:\python help\cnn1.png")
    st.image(image1)