import streamlit as st
from PIL import Image
from model import predict

st.set_page_config(page_title="EfficientNet Image Classifier")

st.title("EfficientNetB0 — Класифікація зображень")
st.write("Завантажте картинку для класифікації за допомогою EfficientNetB0")

uploaded_file = st.file_uploader("Завантажте JPG/PNG", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Завантажене зображення", use_container_width=True)
    
    st.write("Виконується класифікація...")
    preds = predict(image)
    
    st.subheader("Результати:")
    for (class_id, label, prob) in preds:
        st.write(f"**{label}** — {prob*100:.2f}%")