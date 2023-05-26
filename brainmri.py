import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('brainMRI.h5')

categories = ['notumor', 'glioma', 'meningioma', 'pituitary']

# Set a title for your web app
st.title('Brain Tumor MRI Classifier')

# Upload and classify the image
uploaded_file = st.file_uploader('Upload an MRI image', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI', use_column_width=True)

    # Preprocess the image
    image = np.array(image.resize((128, 128))) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    predicted_label = categories[predicted_class]

    # Show the prediction
    st.write('Prediction:', predicted_label)

