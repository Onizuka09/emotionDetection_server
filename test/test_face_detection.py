import cv2
import numpy as np 

WINNAME = "Face Tracking Robot"
WIN_WIDTH = 640
WIN_HEIGHT = 480
DIST_LINES = 40

# Load the Haar cascade file for face detection
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# Create the window and set its size to 300x400
cv2.namedWindow(WINNAME, flags=cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(WINNAME, WIN_WIDTH, WIN_HEIGHT)

while True:
    ret, img = cap.read()  # Capture frame from the default webcam
    if not ret:
        print("Failed to grab frame")
        break
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
  # Draw rectangles around detected faces and their centers
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        center = (int((w / 2) + x), int((h / 2) + y))
        cv2.circle(img, center, 1, (255, 0, 0), 10)
    # Draw vertical lines on the image
    # t = int(WIN_WIDTH / DIST_LINES)
    # for i in range(t):
    #     x_pos = i * DIST_LINES
    #     st_pos = (x_pos, 0)
    #     end_pos = (x_pos, WIN_HEIGHT)
    #     cv2.line(img, st_pos, end_pos, (255, 0, 0), 2)
    # Show the image in the window
    cv2.imshow(WINNAME, img)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
