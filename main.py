import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import functional as F
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.plots import plot_one_box
import numpy as np

# Title and description
st.title("Brain Tumor Detection on MRI")
st.write("This application utilizes YOLOv5 to detect brain tumors on MRI images.")

# Load YOLOv5 model
@st.cache(allow_output_mutation=True)
def load_model():
    return attempt_load("model.pt", map_location=torch.device("cpu")).fuse().eval()

model = load_model()

# Function to process uploaded MRI image
def process_image(image):
    # Resize and convert to tensor
    img = Image.open(image).convert("RGB")
    img = letterbox(img, new_shape=640)[0]
    img = F.normalize(F.resize(img, 640), 0, 255)
    img = torch.unsqueeze(F.to_tensor(img), 0)

    # Predict
    with torch.no_grad():
        detections = model(img)[0]

    # Post-process detections
    detections = non_max_suppression(detections, conf_thres=0.3, iou_thres=0.6)
    if detections[0] is not None:
        for *xyxy, conf, cls in detections[0]:
            plot_one_box(xyxy, img[0], color=(0,255,0), line_thickness=3)
    return img

# Upload MRI image
uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process the image
    processed_img = process_image(uploaded_file)

    # Display processed image
    st.image(processed_img[0].permute(1, 2, 0).numpy().astype(np.uint8), caption='Processed MRI Image', use_column_width=True)
