import pandas as pd
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
def app():
    st.title('Predection of alphabet characters')
    st.write("you can do clasification alphabet characters here")
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
    st.write(
        "link for GUI code  [link](https://github.com/chirag81/"
        "Snapsolution/commit/b226eac802dff1f5338ea00a8b934b279481ecab#diff"
        "-7b462f6fb8d5636379e3c628809b6c00f3e87a81c8e73eb4a7560df7b57fd053)")
    st.write(
        "link for main code [link](https://github.com/chirag81"
        "/Snapsolution/commit/b226eac802dff1f5338ea00a8b934b279481ecab#diff"
        "-dc980fc6165906d1263fbb670fb1ac5d4dd8217d566c7ac8956cb2999ef596fc)")
    word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                 23: 'X', 24: 'Y', 25: 'Z'}

    #df = pd.DataFrame(count,
                      index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                             "S", "T", "U", "V", "W", "X", "Y", "Z"])

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.barh(alphabets, count)


    model = load_model('model_hand.h5')

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

        img_pred = word_dict[np.argmax(model.predict(img_final))]

        st.write(img_pred)


