import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model


def app():
    st.title('Prediction of different flowers')
    st.write("you can do  clasification of flowers here")
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
    st.write(
        "link for GUI code  [link](https://github.com/chirag81"
        "/Snapsolution/commit/b6f86c75e548596e557c660b160b01fa27e8b163#diff"
        "-710efcea92c8d7f574e803cfe7b192e2db41e888778ebc0a5d1d4797b43a2209)")
    st.write(
        "link for main code [link](https://github.com/chirag81/Snapsolution/"
        "commit/b226eac802dff1f5338ea00a8b934b279481ecab#diff-"
        "4d3367809c6bf308e7ba97c03e326de247cf2098eae165804f8085f95b31861e)")
    class_names = ['Daisy',
                   'Sunflower',
                   'Dandelion',
                    'Tulip',
                   'Rose',]
    model = load_model('flowers.h5')
    #shape = ((128, 128, 3))
    #model = tensorflow.keras,hub.KerasLayer(model1,shape)
    #class_f = {'Daisy','Sunflower','tupil','Dandelion','Rose'}
    #model = load_model('earth.h5')
    if image_file is not None:
        image = Image.open(image_file)
        img1 = image.resize((150, 150))
        img1 = preprocessing.image.img_to_array(img1)
        img1 = img1 / 255.0
        img1 = np.expand_dims(img1, axis=0)
        st.image(img1)
       # plt.imshow(image)
        #plt.axis('off')
        #predictions = model.predict(img1)
        pred = class_names[np.argmax(model.predict(img1))]
        #scores = tf.nn.softmax(predictions[0])
        #figure = plt.figure()
        #plt.imshow(image)
        #plt.axis('off')
        #result = predict_class(image)
        #st.write(result)
        #st.pyplot(figure)
        st.write(pred)

