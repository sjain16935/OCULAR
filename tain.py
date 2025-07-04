import os
import cv2
import numpy as np
import pickle

IMAGE_DIR = "register_images"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = []
labels = []
label_id = {}
current_id = 0

for filename in os.listdir(IMAGE_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(IMAGE_DIR, filename)
        label = os.path.splitext(filename)[0]

        if label not in label_id:
            label_id[label] = current_id
            current_id += 1

        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces_detected:
            roi = gray[y:y + h, x:x + w]
            faces.append(roi)
            labels.append(label_id[label])

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

with open("labels.pickle", "wb") as f:
    pickle.dump(label_id, f)

print("Training complete. Model saved as 'trainer.yml' and labels saved as 'labels.pickle'.")
