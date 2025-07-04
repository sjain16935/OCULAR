import cv2
import numpy as np
import os
import pickle
import datetime
import threading
import smtplib
from email.message import EmailMessage
import time

# Flags to control alarm and email
alarm_playing = False
email_sent_for_unknown = False

# ========== ALERT FUNCTIONS ==========

def send_email_alert():
    EMAIL_ADDRESS = "ocular264@gmail.com"       # Your email
    EMAIL_PASSWORD = "aekv dypg hpuw nosq"                    # App password (NOT your login password!)
    TO_EMAIL = "sjain169315@gmail.com"                # Recipient

    msg = EmailMessage()
    msg['Subject'] = "Unknown Person Detected"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL
    msg.set_content("🚨 An unknown person was detected by the face recognition system.")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print("✅ Email alert sent.")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")

def play_alarm():
    global alarm_playing
    try:
        from playsound import playsound
        playsound("alarm.mp3")
    except Exception as e:
        print(f"❌ Failed to play alarm sound: {e}")
    alarm_playing = False  # Reset when done

# ========== HELPER FUNCTIONS ==========

def load_trained_model():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except:
        print("❌ Your OpenCV version may not support face recognizer. Try installing opencv-contrib-python.")
        exit()
    recognizer.read('trainer.yml')
    return recognizer

def log_detection(name):
    log_file_path = "log.txt"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{current_time}: {name}\n"

    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write(log_entry)
    else:
        with open(log_file_path, "r") as f:
            if log_entry not in f.read():
                with open(log_file_path, "a") as fa:
                    fa.write(log_entry)

# ========== MAIN FACE RECOGNITION ==========

def recognize_faces():
    global alarm_playing
    global email_sent_for_unknown

    recognizer = load_trained_model()
    
    with open("labels.pickle", "rb") as f:
        labels = {v: k for k, v in pickle.load(f).items()}

    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("🔍 Starting face recognition... Press 'q' to quit.")
    
    detected_faces = set()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Camera failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        unknown_face_detected = False

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(roi_gray)

            if confidence < 100:
                name = labels.get(label, "Unknown")
            else:
                name = "Unknown"
                unknown_face_detected = True

                # Send email only once
                if not email_sent_for_unknown:
                    threading.Thread(target=send_email_alert).start()
                    email_sent_for_unknown = True

            if name not in detected_faces:
                log_detection(name)
                detected_faces.add(name)

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Alarm logic
        if unknown_face_detected and not alarm_playing:
            alarm_playing = True
            threading.Thread(target=play_alarm).start()

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# ========== START ==========

if __name__ == "__main__":
    recognize_faces()
