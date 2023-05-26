
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('E:/My Files/brainmritest/brainMRI.h5')

# Function to preprocess the image
def preprocess_image(image):
    resized_image = cv2.resize(image, (128, 128))
    preprocessed_image = resized_image / 255.0
    return preprocessed_image

# Function to make predictions
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    class_names = ['notumor', 'glioma', 'meningioma', 'pituitary']
    predicted_class = np.argmax(prediction)
    predicted_label = class_names[predicted_class]
    return predicted_label

# Streamlit app
def main():
    st.title("Brain Tumor Classification")
    st.text("Upload an MRI image and classify the tumor type")

    # Image upload section
    uploaded_file = st.file_uploader("Upload an MRI image", type=['png', 'jpg'])
    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Display the uploaded image
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        # Make predictions
        result = predict(image)

        # Display the prediction result
        if result is not None:
            st.write("Prediction:", result)

if __name__ == "__main__":
    main()
