import joblib
import cv2
import mediapipe as mp_np  # using standard mediapipe import below
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Configure for up to 2 hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)
data = []
labels = []

print("=== MULTI-SIGN ISL DATA COLLECTOR ===")
print("You will record signs one by one.")

while True:
    current_sign_name = (
        input(
            "\nEnter the name of the sign to record (or type 'TRAIN' when done):"
        )
        .strip()
        .upper()
    )

    if current_sign_name == "TRAIN":
        break

    if not current_sign_name:
        print("Please enter a valid sign name.")
        continue

    print(f"\nRecording for: '{current_sign_name}'.")
    print("Look at the camera and hold your sign steady.")
    print("Press SPACEBAR on your keyboard when you have enough samples.")
    print("Press 'q' or ESC to finish recording this specific sign.")

    sample_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)  # Mirror image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Initialize a fresh list for the frame features
        data_aux = []

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract coordinates for each landmark
                for j, lm in enumerate(hand_landmarks.landmark):
                    data_aux.append(lm.x)
                    data_aux.append(lm.y)
                    data_aux.append(lm.z)

        # If only one hand is detected, pad the rest with zeros to maintain a fixed feature size
        expected_length = 126 
        if len(data_aux) < expected_length:
            data_aux.extend([0.0] * (expected_length - len(data_aux)))
        elif len(data_aux) > expected_length:
            data_aux = data_aux[:expected_length]

        # Display feedback on the live OpenCV window
        cv2.putText(
            frame,
            f"Sign: {current_sign_name} | Samples: {sample_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press SPACE to capture, 'q' to next",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        # Press spacebar to save the current frame data
        if key == ord(" "):
            data.append(data_aux)
            labels.append(current_sign_name)
            sample_count += 1
            print(f"Captured sample {sample_count} for {current_sign_name}")
        # Press 'q' to stop recording this sign and enter the next one
        elif key == ord("q") or key == 27:
            print(f"Finished recording {current_sign_name}. Total samples: {sample_count}")
            break

cap.release()
cv2.destroyAllWindows()

# Train the model if data was collected
if len(data) > 0:
    print("\nTraining model with all recorded signs...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(data, labels)
    joblib.dump(model, "isl_model.pkl")
    print(
        f"SUCCESS: New 'isl_model.pkl' saved with signs: "
        f"{list(set(labels))}"
    )
else:
    print("No data recorded.")