import os
import cv2
import mediapipe as mp
import tensorflow as tf

tf.constant(0)

roll_no = input("\nEnter Registration Number: ")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

BASE_DIR = r'C:\Users\ASUS\Documents\sem 8 new\new model\DATA_COL\imgs_without_landmarks'
BASE_DIR_L = r'C:\Users\ASUS\Documents\sem 8 new\new model\DATA_COL'
DATA_DIR = os.path.join(BASE_DIR, roll_no)
LANDMARKS_DIR = os.path.join(BASE_DIR_L, 'imgs with landmark', roll_no)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(LANDMARKS_DIR):
    os.makedirs(LANDMARKS_DIR)

dataset_size = 100

cap = cv2.VideoCapture(1)
while True:
    gesture_name = input("\nEnter gesture name (or type 'quit' to exit): ")
    if gesture_name.lower() == 'quit':
        break

    class_dir = os.path.join(DATA_DIR, gesture_name)
    landmark_class_dir = os.path.join(LANDMARKS_DIR, gesture_name)

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    if not os.path.exists(landmark_class_dir):
        os.makedirs(landmark_class_dir)

    print(f'Collecting data for gesture: {gesture_name}')

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

        img_path = os.path.join(class_dir, f'{counter}.jpg')
        landmark_img_path = os.path.join(landmark_class_dir, f'{counter}.jpg')

        cv2.imwrite(img_path, frame)
        cv2.imwrite(landmark_img_path, frame_with_landmarks)

        cv2.waitKey(25)
        counter += 1

print("Image collection complete!")
cap.release()
cv2.destroyAllWindows()
