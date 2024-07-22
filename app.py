import os
os.environ['LD_LIBRARY_PATH'] = '/path/to/libGL.so.1'
import streamlit as st
import urllib
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

class_labels = ['AI', 'REAL']

def preprocess_image(image):
    image = cv2.resize(image, (32, 32))
    image = image / 255.0  # Normalize image pixel values if needed
    image = image.reshape((1, 32, 32, 3))
    return image

@st.cache_resource
def creator():
    model=load_model("model.h5")
    return model

def predict(image, model):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class = tf.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]
    if confidence <= 0.5:
        predicted_label = 'REAL'
    else:
        predicted_label = 'AI'
    
    return predicted_label

def main():
    st.title("The-Best-Image-Classifier")

    model = creator()
    if model is None:
        st.warning("Please make sure the model file is available.")

    input_method = st.radio("Select input method:", ("Image File", "URL"))

    if input_method == "Image File":
        uploaded_file = st.file_uploader("Upload an image file", type=['jpg', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button('Classify') and model is not None:
                class_label = predict(image, model)
                st.write(f"Class: {class_label}")
                

    elif input_method == "URL":
        url = st.text_input("Enter image URL")
        if url:
            try:
                with urllib.request.urlopen(url) as f:
                    image = Image.open(f)
                    image = np.array(image)
                    st.image(image, caption='Image URL', use_column_width=True)
                    if st.button('Classify') and model is not None:
                        class_label = predict(image, model)
                        st.write(f"Class: {class_label}")
                        
            except urllib.error.URLError:
                st.error("Invalid URL or unable to access the image")

if __name__ == '__main__':
    main()
