import streamlit as st
from PIL import Image

# Set title and description
st.title("Brain Tumor Detection App")
st.write("This app allows you to upload an MRI image for brain tumor detection.")

# Upload MRI image
uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])
