# 1. Place these lines at the absolute top to hide console spam warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 2. Import your required libraries
import streamlit as st
import cv2
import numpy as np

# 3. Streamlit Page Configuration
st.set_page_config(
    page_title="ISL Helper",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 ISL Helper: Indian Sign Language Translator")
st.write("Bridge the gap with real-time sign language recognition.")

# 4. Sidebar controls
st.sidebar.header("Configuration")
app_mode = st.sidebar.selectbox("Choose Mode", ["Live Translation", "About Project"])

if app_mode == "Live Translation":
    st.subheader("Webcam Feed")
    
    # Simple toggle to start/stop the camera feed state
    run_camera = st.checkbox("Turn on Camera")
    
    # Placeholder for translation output
    translation_text_placeholder = st.empty()
    frame_placeholder = st.empty()
    
    # Capture webcam using OpenCV if toggled on
    if run_camera:
        cap = cv2.VideoCapture(0)
        
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam. Please check permissions.")
                break
                
            # Convert frame color format for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # --- [Insert your MediaPipe / ML Prediction Logic Here] ---
            # Example placeholder for translation output loop:
            translated_word = "Translating..." 
            
            # Update UI elements dynamically on every frame loop
            translation_text_placeholder.markdown(f"### Translated Text: **{translated_word}**")
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)
            
        cap.release()
    else:
        st.info("Check the box above to activate your camera and start translating.")

elif app_mode == "About Project":
    st.subheader("About ISL Helper")
    st.write("""
        **ISL Helper** is designed to provide localized, real-time translation for 
        Indian Sign Language (ISL) to bridge communication gaps for the speech- 
        and hearing-impaired community in India.
    """)
