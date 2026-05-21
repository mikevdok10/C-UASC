from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
from flask import Flask, Response
import numpy as np

app = Flask(__name__)

camera = Picamera2()
camera.configure(camera.create_preview_configuration())
camera.start()

model = YOLO('/home/bc/C-UASC/BlankTargetDetector2/runs/detect/train2/weights/best_ncnn_model')

def generate_frames():
    while True:
        frame = camera.capture_array()

        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        labeled_frame = np.ascontiguousarray(frame, dtype=np.uint8)

        

        detectedObjects = model(labeled_frame, conf =0.05 )
    
        labeled_frame = detectedObjects[0].plot()

        

        labeled_frame = cv2.resize(labeled_frame, (500, 500))

    

        
        

        ret, buffer = cv2.imencode('.jpg', labeled_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '<img src="/video_feed">'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='192.168.1.80', port=5000)