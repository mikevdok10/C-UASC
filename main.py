from ultralytics import YOLO

from picamera2 import Picamera2 

import cv2 

camera = Picamera2()
camera.start()

while True:
    frame = camera.capture_array()
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows() 