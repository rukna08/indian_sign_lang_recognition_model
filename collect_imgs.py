import os
import cv2
import mediapipe as mp
import tensorflow as tf

tf.constant(0)

roll_no = input("\nEnter Roll Number: ")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

BASE_DIR = r'C:\Users\ASUS\Documents\sem 8 new\new model\DATA_COL'
DATA_DIR = os.path.join(BASE_DIR, roll_no)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 5  # Change this as needed
dataset_size = 100

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Original Frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        frame_with_landmarks = frame.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_with_landmarks, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Original Frame', frame)
        cv2.imshow('Frame with Landmarks', frame_with_landmarks)

        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        cv2.waitKey(25)
        counter += 1

cap.release()
cv2.destroyAllWindows()
