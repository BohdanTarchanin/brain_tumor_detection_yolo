# main.py
import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load YOLOv5 model
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# Title and description
st.title("Brain Tumor Detection")
st.write("Upload an MRI image to detect brain tumors.")

# Function to load model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = attempt_load(model_path, map_location=torch.device('cpu'))
    return model

# Function to preprocess image
def preprocess_image(image, img_size=480):
    img = Image.open(image)
    img = letterbox(img, new_shape=img_size)[0]
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img

# Function to perform inference
def detect_tumor(image, model):
    img = preprocess_image(image)
    img = img.float() / 255.0  
    img = img if img.ndimension() == 4 else img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.4, 0.5)

    return pred[0] if len(pred) else None

# Upload MRI image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded MRI Image.', use_column_width=True)

    # Load model
    model_path = 'model.pt' 
    model = load_model(model_path)

    # Detect tumor
    tumor_boxes = detect_tumor(uploaded_file, model)

    # Display results
    if tumor_boxes is not None:
        img = Image.open(uploaded_file)
        img = img.convert("RGB")
        img = Image.fromarray((255 * img).astype('uint8'), 'RGB')

        for *box, conf, cls in tumor_boxes:
            box = scale_coords(img.size, box, img.size).round()
            img = img.crop(box)

        st.image(img, caption='Detected Tumor.', use_column_width=True)
    else:
        st.write("No tumor detected.")
