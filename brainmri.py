import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the model
@st.cache(allow_output_mutation=True)
def load_model_from_file():
    model = tf.keras.models.load_model('brainMRI.h5')
    return model

model = load_model_from_file()

categories = ['notumor', 'glioma', 'meningioma', 'pituitary']
class_mapping = {i: category for i, category in enumerate(categories)}

st.write("""
# Brain Tumor Classification
""")
file = st.file_uploader("Upload an MRI image", type=["jpg", "png"])

def preprocess_image(image):
    size = (256, 256)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

if file is None:
    st.text("Please upload an image file")
else:
    try:
        image = Image.open(file)
        image = image.resize((256, 256))  # Resize the image
        st.image(image, use_column_width=True)
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)[0]
        predicted_class = np.argmax(prediction)
        predicted_category = class_mapping[predicted_class]
        confidence = prediction[predicted_class] * 100

        st.write(f"Predicted Category: {predicted_category}")
        st.write(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        st.text("An error occurred while processing the image.")
        st.text(str(e))
