import cv2
import os

def register_user():
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("Error: Could not access the camera.")
        return
    
    user_name = input("Enter your name: ").strip()
    
    if not os.path.exists('register_images'):
        os.makedirs('register_images')
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        cv2.imshow("Register - Press 's' to save, 'q' to quit", frame)
        
        key = cv2.waitKey(1)
        if key % 256 == ord('s'):
            img_path = f"register_images/{user_name}.jpg"
            cv2.imwrite(img_path, frame)
            print(f"Image saved at {img_path}")
            break
        elif key % 256 == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_user()
