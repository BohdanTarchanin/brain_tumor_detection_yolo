import streamlit as st
from PIL import Image

# Set title and description
st.title("Розпізнавання і виявлення патологічних утворень головного мозку")
st.write("Ця програма дозволяє завантажувати зображення МРТ для виявлення пухлини мозку")

# Upload MRI image
uploaded_file = st.file_uploader("Завантажте зображення МРТ", type=["jpg", "jpeg", "png"])
