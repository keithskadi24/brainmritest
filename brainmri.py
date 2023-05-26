import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

model = tf.keras.models.load_model('brainMRI.h5')

def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image)
    image = image.astype('float32') / 255.0
    return image

def main():
    st.title("Brain Tumor MRI Classification")
    st.write("Upload an MRI image to classify the tumor type.")

    # File upload
    uploaded_file = st.file_uploader("Choose an MRI image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(np.expand_dims(image, axis=0))
        predicted_class = np.argmax(predictions[0])
        class_names = ['notumor', 'glioma', 'meningioma', 'pituitary']
        predicted_label = class_names[predicted_class]

        # Display prediction
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)
        st.write("Prediction:", predicted_label)

if __name__ == "__main__":
    main()
