import streamlit as st
from PIL import Image
def app():
    st.title('snap solutions')

    st.write('Step by Step Guide for doing imaga classification')
    st.image("C:\python help\cnn1.png")
    st.subheader('Step 1: Choose a Dataset')
    st.text('Choose a dataset of your interest or you can also create your own ')
    st.text('image dataset for solving your own image classification problem. ')
    st.text('An easy place to choose a dataset is on kaggle.com')
    st.subheader('Step 2: import libraries for the programs')
    st.subheader('Step 3: Prepare Dataset for Training')
    st.text('Preparing dataset for training will involve assigning paths and')
    st.text(' creating categories(labels), resizing images.')
    st.subheader('Step 4: Create Training Data')
    st.text('Training is an array that will contain image pixel values and the index')
    st.text(' at which the image in the CATEGORIES list.')
    st.subheader('Step 5: Shuffle the Dataset')
    st.subheader('Step 6: Assigning Labels and Features')
    st.subheader('step 7: Normalising X and converting labels to categorical data')
    st.subheader('step 8: Split X and Y for use in CNN')
    st.subheader('step 9: define,complie and train the CNN Model')
    st.subheader('step 10: Accuracy and Scope of model')