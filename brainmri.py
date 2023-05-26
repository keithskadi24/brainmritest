import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('brainMRI.h5')

def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def main():
    st.title('Brain Tumor Classifier')
    st.text('Upload an MRI image for tumor classification')

    # File uploader
    uploaded_file = st.file_uploader('Choose an MRI image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Perform inference
        predictions = model.predict(preprocessed_image)
        class_index = np.argmax(predictions[0])
        class_label = categories[class_index]

        # Display the result
        st.image(image, caption=f'Uploaded Image')
        st.write(f'Prediction: {class_label}')

if __name__ == '__main__':
    main()
