import atexit
import io
from fer import FER
import cv2 
from flask import Flask, Response, render_template
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from firebase import init_firebase_connection,set_FirebaseRefrence,set_Value,read_Value
from threading import Condition
from secrets_dir.secrets_pack import databaseURL
import numpy as np
app = Flask(__name__)
detector= FER()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
init_firebase_connection("secrets_dir/emotiondetection-c1441-firebase-adminsdk-igmnb-f0e4b0da40.json",databaseURL)
ref =set_FirebaseRefrence("/emotion")

def emotion_detection(img):
    global ref
    emotion,score = detector.top_emotion(img)
    if score is None:
        score=0
        emotion= " "
    set_Value(score,emotion,ref)
    return emotion,score
# Synchronization and frame buffer class
def detect_face(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
  # Draw rectangles around detected faces and their centers
    emotion,score = emotion_detection(img)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img,f"{emotion}:{score}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
    return img


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

# Initialize camera and configure streaming
picam2 = Picamera2()
# picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
output = StreamingOutput()
# picam2.start_recording(JpegEncoder(), FileOutput(output))
def generate():
    try:
        while True:
            with output.condition:
                output.condition.wait()  # Wait for the next frame to be available
                frame = output.frame
            np_frame = cv2.imdecode(np.frombuffer(frame,np.uint8),cv2.IMREAD_COLOR)
            if np_frame is not None:
                np_frame = detect_face(np_frame)
                _,frame = cv2.imencode('.jpg',np_frame)
                frame = frame.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(e)
@app.before_first_request
def start_camera():
    """Start the camera before serving requests."""
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    picam2.start_recording( JpegEncoder(),FileOutput(output))
@app.route('/home')
def index():
    return render_template('index.html')  # Assumes you have an index.html in the templates directory

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Responds with the MJPEG stream."""
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Gracefully stop the recording when the app exits
# @app.teardown_appcontext
# def stop_camera(exception=None):
#     print("Stooped recording ")
#     picam2.stop_recording()
@atexit.register
def cleanup():
    print("Exiting... ")
    picam2.stop_recording()
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True,debug=False)
