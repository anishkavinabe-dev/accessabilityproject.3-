import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np

# Streamlit Page Configuration
st.set_page_config(
    page_title="ISL Helper",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 ISL Helper: Indian Sign Language Translator")
st.write("Bridge the gap with real-time sign language recognition.")

# Sidebar controls
st.sidebar.header("Configuration")
app_mode = st.sidebar.selectbox("Choose Mode", ["Live Translation", "About Project"])

if app_mode == "Live Translation":
    st.subheader("Webcam Feed")
    st.write("Click **Start** below to activate your browser webcam and begin translation.")

    # Video transformer class for browser-compatible WebRTC streaming
    class ISLTransformer(VideoTransformerBase):
        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            
            # --- [INSERT YOUR MEDIAPIPE OR MODEL PREDICTION LOGIC HERE] ---
            # Example: Drawing text directly onto the video feed frame
            cv2.putText(
                image, 
                text="Translating...", 
                org=(30, 50), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, 
                color=(0, 255, 0), 
                thickness=2
            )
            
            return av.VideoFrame.from_ndarray(image, format="bgr24")

    # WebRTC streamer component replaces OpenCV to work properly on live cloud websites
    webrtc_streamer(
        key="isl-helper-stream",
        video_processor_factory=ISLTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

elif app_mode == "About Project":
    st.subheader("About ISL Helper")
    st.write("""
        **ISL Helper** is designed to provide localized, real-time translation for 
        Indian Sign Language (ISL) to bridge communication gaps for the speech- 
        and hearing-impaired community in India.
    """)
