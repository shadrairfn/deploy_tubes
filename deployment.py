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
def load_tf_model_from_path(model_path):
    return load_model(model_path)

def preprocess_image(img, target_size=(224, 224)):

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
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    return prediction

def display_prediction(prediction, class_names):
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

    st.sidebar.header("📦 Upload Model (.h5)")
    uploaded_model = st.sidebar.file_uploader("Upload model TensorFlow", type=["h5"])
    
    if uploaded_model is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_model:
            tmp_model.write(uploaded_model.read())
            tmp_model_path = tmp_model.name
        model = load_tf_model_from_path(tmp_model_path)
        st.sidebar.success("Model berhasil dimuat dari upload!")
    else:
        st.sidebar.warning("Model belum diupload. Menggunakan model default (jika ada).")
        default_model_path = "final.h5"
        if os.path.exists(default_model_path):
            model = load_tf_model_from_path(default_model_path)
        else:
            st.error("Tidak ada model yang tersedia. Silakan upload model terlebih dahulu.")
            return
    
    uploaded_file = st.file_uploader("Upload gambar sampah", type=["jpg", "jpeg", "png"])
        
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Gambar yang diupload", use_column_width=True)
            
        if st.button("Klasifikasi"):
            with st.spinner("Proses"):
                prediction = predict_image(model, img)
                display_prediction(prediction, class_names)

if __name__ == "__main__":
    main()
