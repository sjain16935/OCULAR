import cv2
import numpy as np
import os

def load_registered_faces():
    images = []
    names = []
    for image_name in os.listdir("register_images"):
        if image_name.endswith(".jpg"):
            img_path = os.path.join("register_images", image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            names.append(image_name[:-4])  # Remove .jpg from name
    return images, names

def recognize_faces():
    # Load the registered faces
    registered_faces, registered_names = load_registered_faces()
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the recognizer
    recognizer.train(registered_faces, np.arange(len(registered_names)))

    # Initialize the webcam for recognition
    cam = cv2.VideoCapture(0)

    print("Starting face recognition... Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            label, confidence = recognizer.predict(roi_gray)

            if confidence < 100:  # Confidence threshold
                name = registered_names[label]
            else:
                name = "Unknown"

            # Draw rectangle around face and put name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)

        # Exit recognition when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
