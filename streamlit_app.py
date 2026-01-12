import streamlit as st
import cv2
import numpy as np
import tempfile
import av

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ===============================
# Streamlit Config
# ===============================
st.set_page_config(
    page_title="YOLOv8 Detection App",
    layout="wide"
)

st.title("🎯 YOLOv8 Detection App (Webcam Ready)")

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
    0.1, 1.0, 0.4, 0.05
)

source_type = st.sidebar.radio(
    "Select Input Type",
    ["Image", "Video", "Webcam (Realtime)"]
)

# =====================================================
# IMAGE DETECTION
# =====================================================
if source_type == "Image":
    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        file_bytes = np.asarray(
            bytearray(uploaded_image.read()),
            dtype=np.uint8
        )
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, channels="BGR", caption="Original")

        if st.button("🔍 Detect"):
            results = model.predict(
                source=image,
                conf=conf_threshold,
                device="cpu"
            )
            annotated = results[0].plot()
            st.image(annotated, channels="BGR", caption="Result")

# =====================================================
# VIDEO DETECTION
# =====================================================
elif source_type == "Video":
    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_area = st.empty()
        stop = st.button("🛑 Stop")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop:
                break

            results = model.predict(
                source=frame,
                conf=conf_threshold,
                device="cpu"
            )

            annotated = results[0].plot()
            frame_area.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

        cap.release()

# =====================================================
# REALTIME WEBCAM (WEBRTC)
# =====================================================
elif source_type == "Webcam (Realtime)":

    st.warning("⚠️ Gunakan Chrome / Edge. Webcam berjalan via WebRTC.")

    class YOLOWebcamProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            results = model.predict(
                source=img,
                conf=conf_threshold,
                device="cpu",
                stream=False
            )

            annotated = results[0].plot()
            return av.VideoFrame.from_ndarray(
                annotated,
                format="bgr24"
            )

    webrtc_streamer(
        key="yolo-webcam",
        video_processor_factory=YOLOWebcamProcessor,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        },
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )
