import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('brainMRI.h5')
    return model

def preprocess_image(image):
    size = (256, 256)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image = np.asarray(image)
    image = image.astype('float32') / 255.0
    return image

def import_and_predict(image, model):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

def main():
    st.title("Brain Tumor Classification")
    image_source = st.sidebar.selectbox("Select Image Source", ("Upload", "URL"))

    if image_source == "Upload":
        file = st.sidebar.file_uploader("Choose a brain MRI image", type=["jpg", "jpeg", "png"])
        if file is None:
            st.sidebar.text("Please upload an image file.")
            return
        image = Image.open(file)

    elif image_source == "URL":
        image_url = st.sidebar.text_input("Enter image URL")
        if image_url == "":
            st.sidebar.text("Please enter an image URL.")
            return
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        except:
            st.sidebar.text("Error: Failed to fetch the image from the provided URL.")
            return

    st.image(image, caption='Uploaded MRI Image.', use_column_width=True)
    model = load_model()
    if st.button("Classify"):
        prediction = import_and_predict(image, model)
        categories = ['notumor', 'glioma', 'meningioma', 'pituitary']
        class_index = np.argmax(prediction)
        class_name = categories[class_index]
        confidence = prediction[0][class_index]
        st.success(f"Predicted Class: {class_name} (Confidence: {confidence:.2f})")

if __name__ == '__main__':
    main()
