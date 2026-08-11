import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Sign Language Translator", page_icon="🤲", layout="centered"
)

st.title("🤲 Real-Time Sign Language Translator")
st.write(
    "Capture a photo of your sign gesture below for instant translation."
)

# Load your trained model
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

# Camera input widget (works perfectly on Render)
camera_image = st.camera_input("Take a picture of your sign")

if camera_image is not None:
    # Convert the uploaded image to an OpenCV array
    bytes_data = camera_image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    cv2_img = cv2.flip(cv2_img, 1)  # Mirror view
    H, W, _ = cv2_img.shape

    image_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
    ) as hands:
        results = hands.process(image_rgb)
        predicted_text = "No hand detected. Try again!"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image
                mp_drawing.draw_landmarks(
                    cv2_img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Extract normalized coordinates
                data_aux = []
                x_ = []
                y_ = []

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                # Run prediction
                if model is not None:
                    try:
                        features = np.asarray(data_aux).reshape(1, -1)

                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(features)
                            confidence = np.max(probabilities)
                            prediction = np.argmax(probabilities)

                            if confidence > 0.60:
                                predicted_text = labels_dict.get(
                                    int(prediction), str(prediction)
                                )
                            else:
                                predicted_text = "Uncertain sign. Please show clearly."
                        else:
                            prediction = model.predict(features)
                            predicted_text = labels_dict.get(
                                int(prediction[0]), str(prediction[0])
                            )
                    except Exception as ex:
                        predicted_text = "Prediction Error"

        # Display result clearly
        st.success(f"### Translation: {predicted_text}")
        
        # Show processed image with drawn hand skeleton
        st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

st.markdown("---")
st.info(
    "💡 **Tips for Best Results:** Ensure good lighting, keep your hand centered"
    " in the frame, and perform a clear sign gesture."
)
