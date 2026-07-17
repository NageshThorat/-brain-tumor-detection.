import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

# Streamlit page configuration
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

st.title("🧠 Brain Tumor Detection from MRI Images")
st.write("Upload an MRI image to detect the presence and type of brain tumor.")

@st.cache_resource
def load_model():
    model_path = 'brain_tumor_model.keras'
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_class_indices():
    class_indices_path = 'class_indices.json'
    if not os.path.exists(class_indices_path):
        return None
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    # Reverse mapping to get index -> class name
    class_names = {v: k for k, v in class_indices.items()}
    return class_names

model = load_model()
class_names = load_class_indices()

if model is None or class_names is None:
    st.warning("Model or class indices not found. Please train the model first by running `python train_model.py` after placing your dataset in the `dataset` folder.")
else:
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display the uploaded image
        st.image(image, caption='Uploaded MRI Image', use_container_width=True)
        
        st.write("Analyzing...")
        
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img_array = preprocess_input(img_array)

        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        predicted_label = class_names[predicted_class_idx].replace('_', ' ').title()
        
        # Display results
        st.success(f"**Prediction:** {predicted_label}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        
        st.write("### Detailed Probabilities:")
        for idx, prob in enumerate(predictions[0]):
            st.write(f"- **{class_names[idx].replace('_', ' ').title()}**: {prob * 100:.2f}%")
