from picamera2 import Picamera2
import picamera2
from ultralytics import YOLO 
import cv2
import numpy as np # computes image arrays of pixels 
model = YOLO("/home/bc/C-UASC/runs_2/detect/train/weights/best_ncnn_model")  # Load the YOLOv8n model
import io
from flask import Flask, Response

app = Flask(__name__)



camera = Picamera2() 
camera.configure(camera.create_still_configuration()) 
camera.start()

def zoom_frame(frame, zoom_factor):
    height, width = frame.shape[:2]
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)
    x1 = (width - new_width) // 2
    y1 = (height - new_height) // 2
    x2 = x1 + new_width
    y2 = y1 + new_height
    zoomed_frame = frame[y1:y2, x1:x2]
    return cv2.resize(zoomed_frame, (width, height))


while True: 
    frame = camera.capture_array() # takes image and stores as an array of pixels
    labeled_frame = np.ascontiguousarray(frame, dtype=np.uint8) # conver to format that yolo can work with 
    detectedObjects = model(labeled_frame, conf=0.05) # returns frame of detected objects 
    
    

    labeled_frame = detectedObjects[0].plot() # this takes the first detection and puts the labels on it 

     # zooms in on the labeled frame
            
    labeled_frame = np.ascontiguousarray(labeled_frame)

    labeled_frame = labeled_frame[:, :, ::-1]  

    labeled_frame = cv2.resize(labeled_frame, (500, 500))


    cv2.imshow("Camera Feed", labeled_frame) # displays labeled frame 
            

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        

camera.stop()
cv2.destroyAllWindows()
        
