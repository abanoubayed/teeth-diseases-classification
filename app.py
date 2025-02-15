import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load the trained Keras model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("teeth_cls_using_mobileNetv2.h5", compile=False)

model = load_model()

st.title("Teeth Disease Classification")
st.write("Welcome! Upload an image of a teeth condition, and our model will analyze it for you.")
uploaded_file = st.file_uploader("choose a image file",type = ["jpg", "png", "jpeg"])

map_dict = {
    0: 'CaS (Canker Sores)',
    1: 'CoS (Cold Sores)',
    2: 'Gum (Gingivostomatitis)',
    3: 'MC (Mouth Cancer)',
    4: 'OC (Oral Cancer)',
    5: 'OLP (Oral Lichen Planus)',
    6: 'OT (Oral Thrush)'
}

target_shape = (224,224)

img_ = ImageDataGenerator(rescale=1 / 255.)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
    cv_img = cv2.imdecode(file_bytes,1)
    cv_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
    resized = cv2.resize(cv_img,target_shape)
    resized = resized / 255.
    img_reshape = resized[np.newaxis, ...]

    st.image(cv_img, channels="RGB")

    generate_pred = st.button("Generate Prediction")
    if generate_pred:
        predictions = model.predict(img_reshape)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100 
        
        st.title("The predicted disease for this image is: {}".format(map_dict [predicted_class]))
        st.write("Confidence: {:.2f}%".format(confidence))

