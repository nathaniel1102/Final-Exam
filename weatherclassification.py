import streamlit as st
import tensorflow as tf
import requests
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('weatherclass.h5')
    return model

model = load_model()
st.write("# Weather Classification")

file_option = st.radio("Choose the source of the image:", ("Upload Image", "URL"))
if file_option == "Upload Image":
    file = st.file_uploader("Choose weather photo from computer", type=["jpg", "png"])
    if file is not None:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_names = ['Cloudy', 'Rain', 'Sunrise', 'Shine']
        string = "OUTPUT: " + class_names[np.argmax(prediction)]
        st.success(string)
elif file_option == "URL":
    image_url = st.text_input("Enter the URL of the image")
    if image_url:
        response = requests.get(image_url)
        try:
            image = Image.open(BytesIO(response.content))
            st.image(image, use_column_width=True)
            prediction = import_and_predict(image, model)
            class_names = ['Cloudy', 'Rain', 'Sunrise', 'Shine']
            string = "OUTPUT: " + class_names[np.argmax(prediction)]
            st.success(string)
        except:
            st.error("Invalid URL or could not fetch the image from the provided URL.")

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction
