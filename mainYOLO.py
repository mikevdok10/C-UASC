from picamera2 import Picamera2
import picamera2
from ultralytics import YOLO 
import cv2
import numpy as np # computes image arrays of pixels 
model = YOLO("yolov8n_ncnn_model")  # Load the YOLOv8n model
import io
from flask import Flask, Response

app = Flask(__name__)



camera = Picamera2() 
camera.configure(camera.create_still_configuration())
camera.start()

cap = cv2.VideoCapture(0)

while True: 
    frame = camera.capture_array() # takes image and stores as an array of pixels
    frame = np.ascontiguousarray(frame, dtype=np.uint8) # conver to format that yolo can work with 
    detectedObjects = model(frame) # returns frame of detected objects 

    labeled_frame = detectedObjects[0].plot() # this takes the first detection and puts the labels on it 
            
    labeled_frame = np.ascontiguousarray(labeled_frame)

    labeled_frame = labeled_frame[:, :, ::-1]  

    labeled_frame = cv2.resize(labeled_frame, (500, 500))


    cv2.imshow("Camera Feed", labeled_frame) # displays labeled frame 
            

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        



camera.stop()
cv2.destroyAllWindows()
        
