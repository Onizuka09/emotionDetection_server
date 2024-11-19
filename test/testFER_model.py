from flask import Flask, render_template, Response
import cv2
import atexit
from fer import FER
# Initialize the camera
camera = cv2.VideoCapture(0)
detector = FER()
# Load Haar Cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if faceCascade.empty():
    raise ValueError("Error loading Haar cascade!")

# Cleanup function to release the camera
def cleanup():
    if camera.isOpened():
        camera.release()
    cv2.destroyAllWindows()

atexit.register(cleanup)

# Mock emotion detection
def emotion_detection(img):
    em,score = detector.top_emotion(img)
    print(em,score)
    if score is None:
        score=0
        em=" "
    return em, score  # Replace with your actual model logic

# Face detection function
def detect_face(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.4, 4)

    if len(faces) == 0:
        print("No faces detected.")
        return img

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Perform emotion detection on the face region
        face_region = img[y:y+h, x:x+w]
        emotion, score = emotion_detection(face_region)
        # Display emotion and score
        if score != 0:
            cv2.putText(img, f"{emotion}:{score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return img

# Main loop to capture video frames
while True:
    success, img = camera.read()
    if not success:
        print("Failed to capture image from camera.")
        break

    processed_img = detect_face(img)
    cv2.imshow("Detected Faces", processed_img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cleanup()

