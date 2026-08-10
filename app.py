import streamlit as st
import numpy as np
import joblib
import mediapipe as mp

# Page configuration
st.set_page_config(page_title="ISL Live Detection", layout="centered")

st.title("ISL Live Sign Language Translation - 2 Hands")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("isl_model.pkl")

try:
    model = load_model()
    st.success(f"Model loaded successfully! Model expects {model.n_features_in_} features.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

# Streamlit UI elements
run_webcam = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])
prediction_text = st.empty()

cap = cv2.VideoCapture(0)

while run_webcam:
    success, frame = cap.read()
    if not success:
        st.warning("Failed to access webcam.")
        break

    # Flip the frame for a natural mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    data_aux = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            
            # Extract x, y, z coordinates
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x)
                data_aux.append(hand_landmarks.landmark[i].y)
                data_aux.append(hand_landmarks.landmark[i].z)

        # Pad with zeros if only 1 hand is detected to match model features
        if len(results.multi_hand_landmarks) == 1:
            data_aux = data_aux + [0.0] * (126 - len(data_aux))

        if len(data_aux) == model.n_features_in_:
            landmarks_np = np.array(data_aux).reshape(1, -1)
            prediction = model.predict(landmarks_np)
            pred_label = f"Prediction: {prediction[0]}"
            prediction_text.markdown(f"## {pred_label}")
        else:
            prediction_text.markdown("## Align hands properly in view")

    # Display the video frame in Streamlit
    FRAME_WINDOW.image(frame, channels="BGR")

else:
    cap.release()
    prediction_text.markdown("## Webcam is stopped.")
