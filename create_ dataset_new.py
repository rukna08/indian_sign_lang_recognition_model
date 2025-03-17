import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
accepted_count = 0
rejected_count = 0

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            print(f"Rejected (cannot read): {dir_}/{img_path}")
            rejected_count += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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

            # Ensure each sample has a fixed length by padding if needed
            while len(data_aux) < 84:  # 21 landmarks * 2 coordinates * 2 hands
                data_aux.append(0.0)

            data.append(data_aux)
            labels.append(dir_)
            accepted_count += 1
            print(f"Accepted: {dir_}/{img_path}")
        else:
            print(f"Rejected (no hand detected): {dir_}/{img_path}")
            rejected_count += 1

print(f"\nTotal images accepted: {accepted_count}")
print(f"Total images rejected: {rejected_count}")

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
