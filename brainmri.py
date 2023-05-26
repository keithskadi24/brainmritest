import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('brainMRI.h5')

@tf.function
def preprocess_image(image_file):
    img = np.array(Image.open(image_file))
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main():
    st.title("Brain Tumor Classification")
    st.write("Upload an MRI image for tumor classification.")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the image
        img = preprocess_image(uploaded_file)

        # Make predictions
        predictions = model.predict(img)
        class_names = ['notumor', 'glioma', 'meningioma', 'pituitary']
        predicted_class = class_names[np.argmax(predictions)]

        # Display the result
        st.image(uploaded_file, caption='Uploaded MRI', use_column_width=True)
        st.success(f"Predicted Class: {predicted_class}")

if __name__ == '__main__':
    main()

