import cv2
import numpy as np
import os
import pickle
import datetime
import smtplib
from email.message import EmailMessage

def send_email_alert():
    EMAIL_ADDRESS = "shubham.16935@sakec.ac.in"       # Replace with your email
    EMAIL_PASSWORD = "mahavir@123"         # Replace with your app password (not normal password)
    TO_EMAIL = "Sjain169315@gmail.com"     # Where you want to send the alert

    msg = EmailMessage()
    msg['Subject'] = "Unknown Person Detected"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL
    msg.set_content("An unknown person has been detected by the face recognition system.")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("🚨 Email alert sent!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


def load_trained_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    return recognizer

def log_detection(name):
    if not os.path.exists("log.txt"):
        with open("log.txt", "w") as log_file:
            pass

    with open("log.txt", "r") as log_file:
        log_data = log_file.read()

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{current_time}: {name}\n"
    if log_entry not in log_data:
        with open("log.txt", "a") as log_file:
            log_file.write(log_entry)

def recognize_faces():
    recognizer = load_trained_model()
    
    with open("labels.pickle", "rb") as f:
        labels = {v: k for k, v in pickle.load(f).items()}

    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Starting face recognition... Press 'q' to quit.")
    
    detected_faces = set()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            label, confidence = recognizer.predict(roi_gray)

            if confidence < 100:
                name = labels[label]
            else:
                name = "Unknown"

            if name not in detected_faces:
                log_detection(name)
                detected_faces.add(name)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()