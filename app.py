import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np

# Load model
model = tf.keras.models.load_model("model.h5")

st.title("🐶🐱 Cat vs Dog Classifier")

# Upload file
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file is not None:
    try:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image")

        img = image.resize((128,128))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)

        if prediction[0][0] > 0.5:
            st.success("🐶 Dog")
        else:
            st.success("🐱 Cat")

    except UnidentifiedImageError:
        st.error("❌ Please upload a valid image (JPG/PNG)")