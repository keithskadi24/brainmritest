import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('brainMRI.h5')
    return model

model = load_model()
categories = ['notumor', 'glioma', 'meningioma', 'pituitary']

st.title("Brain Tumor MRI Classification")
file = st.file_uploader("Choose a brain MRI image", type=["jpg", "png"])

def preprocess_image(image):
    resized_image = image.resize((128, 128))
    normalized_image = np.array(resized_image) / 255.0
    reshaped_image = np.reshape(normalized_image, (1, 128, 128, 3))
    return reshaped_image

def classify_image(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    class_index = np.argmax(prediction)
    class_name = categories[class_index]
    confidence = prediction[0][class_index]
    return class_name, confidence

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    class_name, confidence = classify_image(image, model)
    st.subheader("Prediction:")
    st.write(f"Class: {class_name}")
    st.write(f"Confidence: {confidence:.2f}")
