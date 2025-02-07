import cv2
import mediapipe as mp
import pandas as pd
import time
import os

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)

# Open webcam.
cap = cv2.VideoCapture(0)

# Data will hold rows of [label, hand0_x0, hand0_y0, ..., hand0_x20, hand0_y20, hand1_x0, hand1_y0, ..., hand1_x20, hand1_y20].
data = []

debug = 1

while True:
    label = None
    if not debug:
        label = input("Enter label for this gesture (or 'exit' to stop): ")
        if label.lower() == 'exit':
            break

    if debug:
        label = "0"

    # Create a folder for the label if it doesn't exist.
    folder_path = f"dataset/{label}"
    os.makedirs(folder_path, exist_ok=True)

    if not debug:
        print("Prepare for the gesture. Starting in 10 seconds...")
        time.sleep(10)

    count = 0
    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for a more intuitive experience.
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe Hands.
        results = hands.process(rgb_frame)

        # Initialize landmarks_list for two hands (42 values per hand: x and y for 21 landmarks each).
        landmarks_list = [None] * (42 * 2)

        if results.multi_hand_landmarks:
            # Process each detected hand.
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for j, landmark in enumerate(hand_landmarks.landmark):
                    # Store coordinates in alternating order: x0, y0, x1, y1, ..., x20, y20.
                    index = i * 42 + j * 2
                    landmarks_list[index] = landmark.x
                    landmarks_list[index + 1] = landmark.y

                # Draw the hand landmarks on the frame.
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Append the label and the landmarks to our dataset.
            data.append([label] + landmarks_list)

            # --- Overlay the extreme left hand's coordinates on the frame ---
            # Determine which detected hand is farthest to the left.
            extreme_left_hand = min(
                results.multi_hand_landmarks,
                key=lambda hand: min(lm.x for lm in hand.landmark)
            )

            # Text Rendering.
            if debug:
                # Draw the 21 landmark coordinates for the extreme left hand.
                start_y = 20  # Vertical starting position for text.
                for idx, lm in enumerate(extreme_left_hand.landmark):
                    text = f"Point {idx}: x: {lm.x:.2f}, y: {lm.y:.2f}"

                    cv2.putText(frame, text, (10, start_y + idx * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, text, (10, start_y + idx * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

                # ---------------------------------------------------------------

            # Save the annotated frame (with overlays) to disk.
            img_path = os.path.join(folder_path, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            count += 1
            print(f"Picture {count} done")
            time.sleep(0.1)  # Delay to slow down capture.

        # Show the frame with overlays.
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Collected 100 images for label: {label}")

# --- Save the Dataset to Excel ---
# Build header to match the data order:
# [label, hand0_x0, hand0_y0, hand0_x1, hand0_y1, ..., hand0_x20, hand0_y20, hand1_x0, hand1_y0, ..., hand1_x20, hand1_y20]
header = ["label"]
for hand in range(2):
    for i in range(21):
        header.append(f"hand{hand}_x{i}")
        header.append(f"hand{hand}_y{i}")

# Create a DataFrame and write to an Excel file.
df = pd.DataFrame(data, columns=header)
excel_filename = "dataset_labels.xlsx"
df.to_excel(excel_filename, index=False)

print(f"Dataset saved as {excel_filename}")

# Release resources.
cap.release()
cv2.destroyAllWindows()
