
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model


def app():
    st.title('Prediction of traffic signs')
    st.write("you can do clasification of different traffic signs here")

    image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
    st.write(
        "link for GUI code  [link](https://github.com/chirag81/snapsolution1/blob/main/traffic_signs.py)")
    st.write(
        "link for main code [link](https://github.com/chirag81/snapsolution1/blob/main/traffic_signs1.py)")
   # img1 = st.file_uploader("Please upload an brain scan file", type=["jpg","png"])
    # load the trained model to classify traffic signs

    # model = load_model('traffic_classifier.h5')
    # dictionary to label all traffic signs class.
    classe1 = {0: 'Speed limit (20km/h)',
               1: 'Speed limit (30km/h)',
               2: 'Speed limit (50km/h)',
               3: 'Speed limit (60km/h)',
               4: 'Speed limit (70km/h)',
               5: 'Speed limit (80km/h)',
               6: 'End of speed limit (80km/h)',
               7: 'Speed limit (100km/h)',
               8: 'Speed limit (120km/h)',
               9: 'No passing',
               10: 'No passing veh over 3.5 tons',
               11: 'Right-of-way at intersection',
               12: 'Priority road',
               13: 'Yield',
               14: 'Stop',
               15: 'No vehicles',
               16: 'Veh > 3.5 tons prohibited',
               17: 'No entry',
               18: 'General caution',
               19: 'Dangerous curve left',
               20: 'Dangerous curve right',
               21: 'Double curve',
               22: 'Bumpy road',
               23: 'Slippery road',
               24: 'Road narrows on the right',
               25: 'Road work',
               26: 'Traffic signals',
               27: 'Pedestrians',
               28: 'Children crossing',
               29: 'Bicycles crossing',
               30: 'Beware of ice/snow',
               31: 'Wild animals crossing',
               32: 'End speed + passing limits',
               33: 'Turn right ahead',
               34: 'Turn left ahead',
               35: 'Ahead only',
               36: 'Go straight or right',
               37: 'Go straight or left',
               38: 'Keep right',
               39: 'Keep left',
               40: 'Roundabout mandatory',
               41: 'End of no passing',
               42: 'End no passing veh > 3.5 tons'}
    model = load_model('traffic_classifier.h5')
    #img1 = cv2.imread(r"C:\python help\traffic\Meta\2.png")
    #plt.imshow(img1)
    #plt.show()
    if image_file is not None:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR")
        #img = Image.open(img1)
        img = cv2.resize(image,(30,30))
        st.image(img)
    # img = image.load_img(img1, target_size=(30, 30))
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
    # pred = model.predict(img4)
        pred = classe1[np.argmax(model.predict(img))]
    # pred = classes1[model.predict(img4)]
    # pred = model.predict_classes([image])[0]
        st.write(pred)



