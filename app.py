import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp
from PIL import Image

# Page Configuration
st.set_page_config(page_title="ISL Live Sign Language Translation", layout="centered")

st.title("ISL Live Sign Language Translation - 2 Hands")

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

# Streamlit Cloud compatible camera input widget
img_file = st.camera_input("Take a snapshot of your sign language gesture")

if img_file is not None:
    # Convert uploaded image buffer to a numpy array
    image = Image.open(img_file)
    frame = np.array(image)
    
    # Process frame with MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        results = hands.process(frame)
        
        if results.multi_hand_landmarks:
            hand_data = []
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image for visual feedback
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                single_hand_features = []
                for landmark in hand_landmarks.landmark:
                    single_hand_features.extend([landmark.x, landmark.y, landmark.z])
                hand_data.append(single_hand_features)
            
            # Flatten features for all detected hands
            flat_features = []
            for h_feats in hand_data:
                flat_features.extend(h_feats)
            
            # Ensure the feature vector is strictly 126 elements (pad with zeros if only 1 hand is present)
            if len(flat_features) < 126:
                flat_features.extend([0.0] * (126 - len(flat_features)))
            elif len(flat_features) > 126:
                flat_features = flat_features[:126]
                
            features = np.array(flat_features).reshape(1, -1)
            
            # Display the frame with drawn hand landmarks
            st.image(frame, channels="RGB", caption="Processed Hand Landmarks")
            
            # Run prediction through the loaded model
            if model is not None:
                try:
                    prediction = model.predict(features)
                    st.markdown(f"### Predicted Sign: **{prediction[0]}**")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.warning("No hands detected in the frame. Please make sure both hands are clearly visible and try again.")
