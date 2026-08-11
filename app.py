import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Page Configuration
st.set_page_config(
    page_title="ISL Helper - Sign Language Translator", 
    page_icon="🤲", 
    layout="centered"
)

# Sidebar Navigation
app_mode = st.sidebar.selectbox("Choose Mode", ["Live Translation", "About Project"])

if app_mode == "Live Translation":
    st.title("🤲 Real-Time Sign Language Translator")
    st.write(
        "Allow camera access to stream and translate sign language gestures live."
    )

    # Load your trained model
    @st.cache_resource
    def load_model():
      try:
        model = joblib.load("isl_model.pkl")
        return model
      except Exception as e:
        return None

    model = load_model()

    # Initialize MediaPipe Hands for up to 2 hands (126 features total)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles


    class SignLanguageProcessor:

      def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

      def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        predicted_text = "Show sign clearly..."

        if results.multi_hand_landmarks:
          data_aux = []

          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            for landmark in hand_landmarks.landmark:
              data_aux.append(landmark.x)
              data_aux.append(landmark.y)
              data_aux.append(landmark.z)

          # Pad with zeros if only 1 hand is detected to match the 126 features requirement
          while len(data_aux) < 126:
            data_aux.append(0.0)

          # Trim if excess features are present
          data_aux = data_aux[:126]

          if model is not None:
            try:
              features = np.asarray(data_aux).reshape(1, -1)
              
              # Directly predict text labels without relying on number IDs
              prediction = model.predict(features)
              predicted_text = str(prediction[0])
              
            except Exception as ex:
              predicted_text = "Translating..."

        cv2.putText(
            image,
            f"Sign: {predicted_text}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


    # Streamlit WebRTC Live Streamer with Custom TURN Server and Unique Key
    webrtc_streamer(
        key="sign-language-translator-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignLanguageProcessor,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": [
                        "turn:global.relay.metered.ca:80",
                        "turn:global.relay.metered.ca:443",
                    ],
                    "username": "be800e03744f0a40a274fdab",
                    "credential": "wRRR0DZ7be8ghbP6",
                },
            ]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

    st.markdown("---")
    st.info(
        "💡 **Tips for Best Results:** Ensure good lighting and keep your hands"
        " centered within the camera frame."
    )

elif app_mode == "About Project":
    st.title("🤟 About ISL Helper")
    st.write("""
        **ISL Helper** is designed to provide localized, real-time translation for 
        Indian Sign Language (ISL) to bridge communication gaps for the speech- 
        and hearing-impaired community in India.
    """)
