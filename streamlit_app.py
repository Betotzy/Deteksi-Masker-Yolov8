import time

# ===============================
# VIDEO DETECTION (SMOOTH & ANTI FREEZE)
# ===============================
elif source_type == "Video":
    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        import time

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)

        stframe = st.empty()
        stop_button = st.button("🛑 Stop")

        frame_count = 0
        DETECT_EVERY_N = 5        # 🔥 YOLO tiap 5 frame
        last_annotated = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_button:
                break

            frame_count += 1

            # ⚡ resize dulu (WAJIB)
            frame = cv2.resize(frame, (640, 640))

            # 🔍 YOLO hanya tiap N frame
            if frame_count % DETECT_EVERY_N == 0:
                results = model(
                    frame,
                    conf=conf_threshold,
                    device="cpu",
                    verbose=False
                )
                last_annotated = results[0].plot()

            # 🧠 tampilkan frame terakhir agar video tetap jalan
            stframe.image(
                last_annotated if last_annotated is not None else frame,
                channels="BGR",
                use_container_width=True
            )

            # ⏱️ kecil tapi penting untuk stabilitas UI
            time.sleep(0.01)

        cap.release()
