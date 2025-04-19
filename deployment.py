import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import tempfile
import time
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Klasifikasi Sampah - TensorFlow", layout="wide")

@st.cache_resource
def load_tf_model():
    """Load trained TensorFlow model"""
    model_path = "D:\\ML & DL\\Sampah\\final.h5"
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan di {model_path}. Silakan upload model terlebih dahulu.")
        return None
    return load_model(model_path)

def preprocess_image(img, target_size=(224, 224)):
    """Preprocess image for model prediction"""
 
    img = cv2.resize(img, target_size)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4: 
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 3:  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(model, img):
    """Make prediction using the model"""
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    return prediction

def display_prediction(prediction, class_names):
    """Display prediction results"""
    class_idx = np.argmax(prediction[0])
    confidence = float(prediction[0][class_idx])
    
    st.write(f"### Hasil Klasifikasi: ")
    if confidence > 0.5:
        st.write("### Anorganik")
    else:
        st.write("### Organik")

def main():
    st.title("Klasifikasi Sampah Organik dan Anorganik")
    st.write("Aplikasi ini mengklasifikasikan jenis sampah menggunakan model TensorFlow")
    
    class_names = ['Organik', 'Anorganik']
    
    if 'model_path' in st.session_state:
        model = load_model(st.session_state['model_path'])
        st.sidebar.success("Menggunakan model yang diupload")
    else:
        model = load_tf_model()
        if model is None:
            st.warning("Model tidak ditemukan. Silakan upload model terlebih dahulu.")
            return
    
 
    uploaded_file = st.file_uploader("Upload gambar sampah", type=["jpg", "jpeg", "png"])
        
    if uploaded_file is not None:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Gambar yang diupload", use_column_width=True)
            
        if st.button("Klasifikasi"):
            with st.spinner("Memproses..."):
                prediction = predict_image(model, img)
                display_prediction(prediction, class_names)
    

if __name__ == "__main__":
    main()