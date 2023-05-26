import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('brainMRI.h5')
    return model

model = load_model()

st.title("Brain Tumor Classifier")
file = st.file_uploader("Choose a brain MRI image", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@tf.function
def predict(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    prediction = predict(image, model)
    class_names = ['notumor', 'glioma', 'meningioma', 'pituitary']
    predicted_class = class_names[np.argmax(prediction)]
    st.success(f"Predicted Class: {predicted_class} (Confidence: {prediction[0][np.argmax(prediction)]:.4f})")
