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
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 25

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 640))

        stframe = st.empty()
        stop_button = st.button("🛑 Stop")

        frame_count = 0
        DETECT_EVERY_N = 5
        last_annotated = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                break

            frame = cv2.resize(frame, (640, 640))
            frame_count += 1

            if frame_count % DETECT_EVERY_N == 0:
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    device="cpu",
                    verbose=False
                )
                last_annotated = results[0].plot()

            output_frame = last_annotated if last_annotated is not None else frame

            stframe.image(output_frame, channels="BGR", use_container_width=True)
            out.write(output_frame)

            time.sleep(0.01)

        cap.release()
        out.release()

        st.success("✅ Video selesai diproses")

        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Output Video",
                data=f,
                file_name="output_detection.mp4",
                mime="video/mp4"
            )

# ===============================
# REALTIME WEBCAM (DESKTOP & MOBILE)
# ===============================
elif source_type == "Webcam":
    st.subheader("📷 Realtime Webcam Detection")

    class YOLOVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (640, 480))

            results = model.predict(
                source=img,
                conf=conf_threshold,
                device="cpu",
                imgsz=640,
                verbose=False
            )

            annotated = results[0].plot()

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="yolo-webcam",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={
            "video": {
                "facingMode": "environment",
                "width": {"ideal": 640},
                "height": {"ideal": 480}
            },
            "audio": False
        },
        async_processing=True
    )
