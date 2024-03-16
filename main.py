import streamlit as st
import torch
from PIL import Image
import requests
from io import BytesIO

# Load YOLOv5 dependencies
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.plots import plot_one_box

# Title and short description
st.title("Brain Tumor Detection App")
st.write("This app uses YOLOv5 to detect brain tumors on MRI images.")

# Function to process image with YOLOv5
@st.cache(allow_output_mutation=True)
def detect_tumor(image):
    # Load YOLOv5 model
    model = attempt_load("model.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Resize image
    img = letterbox(image, new_shape=640)[0]

    # Convert image to tensor
    img = torch.from_numpy(img).to(torch.device('cuda')).unsqueeze(0).float() / 255.0

    # Inference
    pred = model(img, augment=False)[0]

    # Post-processing
    pred = non_max_suppression(pred, conf_thres=0.4)

    # Draw bounding boxes
    for det in pred:
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
        for *xyxy, conf, cls in reversed(det):
            label = f'{model.names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, image, label=label, color=(0, 255, 0), line_thickness=3)

    return image

# File upload
uploaded_file = st.file_uploader("Upload MRI image", type=['jpg', 'jpeg', 'png'])

# Process uploaded image and display
if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    
    # Detect tumor
    output_image = detect_tumor(image)

    # Display processed image
    st.image(output_image, caption='Processed Image', use_column_width=True)
