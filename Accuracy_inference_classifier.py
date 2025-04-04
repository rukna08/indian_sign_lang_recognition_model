import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
               'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H',
               'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P',
               'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X',
               'Y': 'Y', 'Z': 'Z', 'SPACE': 'SPACE', 'NEXT': 'NEXT', 'DONE': 'DONE'}

gesture_accuracies = {}
#
while True:
    gesture_name = input("Enter gesture name (or 'END' to finish): ").strip()
    if gesture_name == "END":
        break

    print("Starting in 5 seconds...")
    time.sleep(5)

    correct_predictions = 0
    total_samples = 100

    for _ in range(total_samples):
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            while len(data_aux) < 84:
                data_aux.append(0.0)

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(prediction[0], "Unknown")

            if isinstance(prediction[0], (int, np.integer)):
                predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
            elif prediction[0] in labels_dict.values():
                predicted_character = prediction[0]
            else:
                predicted_character = "Unknown"

            print(f"Predicted Raw Output: {prediction[0]}")

            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                        cv2.LINE_AA)

            if predicted_character == gesture_name:
                correct_predictions += 1

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    accuracy = (correct_predictions / total_samples) * 100
    gesture_accuracies[gesture_name] = accuracy
    print(f"Accuracy for {gesture_name}: {accuracy:.2f}%")

cap.release()
cv2.destroyAllWindows()

print("\nFinal Gesture Accuracies:")
for gesture, acc in gesture_accuracies.items():
    print(f"{gesture}: {acc:.2f}%")