import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av

# Page Configuration
st.set_page_config(
    page_title="Sign Language Translator", page_icon="🤲", layout="centered"
)

st.title("🤲 Real-Time Sign Language Translator")
st.write(
    "Allow camera access to translate sign language gestures in real-time."
)

# Load your trained model (Make sure your model file is in the repo)
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.p")
        return model
    except Exception as e:
        return None

model = load_model()

# Define your label mapping dictionary (Modify these based on your training classes)
labels_dict = {
    0: "Hello",
    1: "Thank You",
    2: "Yes",
    3: "No",
    # Add all your trained class mappings here
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class SignLanguageProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)  # Flip horizontally for natural mirror view
        H, W, _ = image.shape

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        predicted_text = "Show sign clearly..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the video frame
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Extract normalized landmark coordinates
                data_aux = []
                x_ = []
                y_ = []

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                # Predict if model is loaded successfully
                if model is not None:
                    try:
                        features = np.asarray(data_aux).reshape(1, -1)

                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(features)
                            confidence = np.max(probabilities)
                            prediction = np.argmax(probabilities)

                            if confidence > 0.75:
                                predicted_text = labels_dict.get(
                                    int(prediction), str(prediction)
                                )
                            else:
                                predicted_text = "Translating..."
                        else:
                            prediction = model.predict(features)
                            predicted_text = labels_dict.get(
                                int(prediction[0]), str(prediction[0])
                            )
                    except Exception as ex:
                        predicted_text = "Prediction Error"

        # Overlay translation text directly onto the video feed
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


# Streamlit WebRTC Component with integrated public STUN servers
webrtc_streamer(
    key="sign-language-stream",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignLanguageProcessor,
    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302",
                    "stun:stun1.l.google.com:19302",
                    "stun:stun.freebuzzer.com:3478",
                ]
            }
        ]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
st.info(
    "💡 **Tips for Best Results:** Ensure good lighting, keep your hand centered"
    " within the camera frame, and make distinct sign gestures matching your"
    " training data."
)
