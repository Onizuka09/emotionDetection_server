from flask import Flask,render_template,Response
from fer import FER
import cv2 
import atexit
from threading import Thread,Lock
app = Flask(__name__)
detector= FER()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0);
fram_lock = Lock()
output_frame=None
def emotion_detection(img):
    emotion,score = detector.top_emotion(img)
    if score is None:
        score=0
        emotion= " "
    return emotion,score
def detect_face(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
  # Draw rectangles around detected faces and their centers
    emotion,score = emotion_detection(img)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img,f"{emotion}:{score}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    return img
def capture_frames():
    global output_frame
    while  True:
        succ, frame = camera.read()
        if not succ:
            camera.release()
            print("Failed to Capture frame")
            break; 
        frame = detect_face(frame)
        with fram_lock:
            output_frame=frame
thread = Thread(target=capture_frames,daemon=True)
thread.start()
def image_capture():
    camera = cv2.VideoCapture(0);
    while True:
        with fram_lock:
            if output_frame is None:
                continue
            
        ret,buff = cv2.imencode('.jpg', frame)
        frame = buff.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
    camera.release()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(image_capture(),mimetype='multipart/x-mixed-replace; boundary=frame')
# @atexit.register
# def cleanup():
#     camera.release()
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',threaded=True)
