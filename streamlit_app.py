import streamlit as st
import cv2
import tempfile
import numpy as np
import av
import time

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ===============================
# Streamlit Config
# ===============================
st.set_page_config(
    page_title="YOLOv8 Detection App",
    layout="wide"
)

st.title("🎯 YOLOv8 Image, Video & Realtime Webcam Detection")

# ===============================
# Load Model (cached)
# ===============================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ===============================
# Sidebar
# ===============================
st.sidebar.header("⚙️ Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

source_type = st.sidebar.radio(
    "Select Input Type",
    ["Image", "Video", "Webcam"]
)

# ===============================
# IMAGE DETECTION
# ===============================
if source_type == "Image":
    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.subheader("Original Image")
        st.image(image, channels="BGR", use_container_width=True)

        if st.button("🔍 Detect Objects"):
            results = model.predict(
                source=image,
                conf=conf_threshold,
                device="cpu",
                verbose=False
            )

            annotated = results[0].plot()

            st.subheader("Detection Result")
            st.image(annotated, channels="BGR", use_container_width=True)

# ===============================
# VIDEO DETECTION + DOWNLOAD
# ===============================
elif source_type == "Video":
    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(
