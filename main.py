import streamlit as st
import os
import shutil
import zipfile
import urllib.request
import torch
import subprocess
from PIL import Image

# Global constants
OUTPUT_FOLDER = 'output_models'

def download_and_unzip_data():
    if not os.path.isfile('data.zip'):
        urllib.request.urlretrieve("https://github.com/giuseppebrb/BrainTumorDetection/blob/main/data.zip?raw=true", "data.zip")

    with zipfile.ZipFile("data.zip", "r") as zip_ref:
        zip_ref.extractall(".")

    os.remove('data.zip')

def train_model():
    if torch.cuda.is_available():
        device = 0
    else:
        device = 'cpu'

    # Training axial plane
    subprocess.run(['python', 'yolov5/train.py', '--img', '480', '--batch', '64', '--epochs', '400', '--data', './data/axial/axial.yaml', '--weights', 'yolov5m.pt', '--device', str(device), '--name', 'axial', '--hyp', './data/augmentation.yaml'])

    # Copy the fine-tuned model inside the output folder
    shutil.copyfile('yolov5/runs/train/axial/weights/best.pt', f'{OUTPUT_FOLDER}/tumor_detector_axial.pt')

def detect_tumor(image):
    # Run detection
    subprocess.run(['python', 'yolov5/detect.py', '--weights', 'output_models/tumor_detector_axial.pt', '--img', '640', '--conf', '0.4', '--source', image, '--save-txt'])

    # Display result
    image_path = 'yolov5/runs/detect/exp/b510dc0d5cd3906018c4dd49b98643_gallery.jpeg'
    result_image = Image.open(image_path)
    st.image(result_image, caption='Tumor detection result', use_column_width=True)

def main():
    st.title('Brain Tumor Detection App')

    # Download and unzip data
    download_and_unzip_data()

    # Train model
    train_model()

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        detect_tumor(uploaded_image)

if __name__ == '__main__':
    main()
