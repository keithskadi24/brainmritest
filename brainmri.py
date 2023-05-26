import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from tensorflow import keras

@st.cache(allow_output_mutation=True)
def load_model():
    model = keras.models.load_model('brainMRI.h5')
    return model

def import_and_predict(image_data, model):
    size = (128, 128)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = np.asarray(image)
    image = image / 255.0
    img_reshape = np.reshape(image, (1, 128, 128, 3))
    prediction = model.predict(img_reshape)
    return prediction

def main():
    st.title("Brain Tumor Classification")
    file = st.file_uploader("Choose a brain MRI image", type=["jpg", "jpeg", "png"])

    if file is None:
        st.text("Please upload an image file.")
    else:
        image = Image.open(file)
        st.image(image, caption='Uploaded MRI Image.', use_column_width=True)
        model = load_model()
        if st.button("Classify"):
            prediction = import_and_predict(image, model)
            categories = ['notumor', 'glioma', 'meningioma', 'pituitary']
            class_index = np.argmax(prediction)
            class_name = categories[class_index]
            st.success("Predicted Class: {}".format(class_name))

if __name__ == '__main__':
    main()
