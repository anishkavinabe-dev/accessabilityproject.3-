import streamlit as st
import av
import cv2
import numpy as np
import joblib
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Page Configuration
st.set_page_config(page_title="ISL Live Sign Language Translation", layout="centered")

st.title("ISL Live Sign Language Translation - Continuous Real-Time")

# Load Model Safely
@st.cache_resource
def load_model():
    try:
        model = joblib.load('isl_model.pkl')
        return model
    except Exception as e:
        return None

model = load_model()

if model is not None:
    st.success("Model loaded successfully! Model expects 126 features.")
else:
    st.error("Error loading model: 'isl_model.pkl' not found.")

# Initialize MediaPipe Hands (Max 2 hands)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define Video Processor Class for Continuous Streaming
class HandProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Convert BGR to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            hand_data = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                single_hand_features = []
                for landmark in hand_landmarks.landmark:
                    single_hand_features.extend([landmark.x, landmark.y, landmark.z])
                hand_data.append(single_hand_features)
            
            # Flatten features for all detected hands
            flat_features = []
            for h_feats in hand_data:
                flat_features.extend(h_feats)
            
            # Ensure the feature vector is strictly 126 elements (pad with zeros if needed)
            if len(flat_features) < 126:
                flat_features.extend([0.0] * (126 - len(flat_features)))
            elif len(flat_features) > 126:
                flat_features = flat_features[:126]
                
            features = np.array(flat_features).reshape(1, -1)
            
            # Run prediction through the model and overlay text onto the live video feed
            if model is not None:
                try:
                    prediction = model.predict(features)
                    cv2.putText(
                        img, 
                        f"Sign: {prediction[0]}", 
                        (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2, 
                        cv2.LINE_AA
                    )
                except Exception as e:
                    pass
                    
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start WebRTC Live Stream Component
webrtc_streamer(
    key="isl-translation",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=HandProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
