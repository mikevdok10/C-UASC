
from picamera2 import Picamera2 
from picamera2.encoders import MJPEGEncoder
import time 


import cv2 

camera = Picamera2()

config = camera.create_preview_configuration()
camera.configure(config)
camera.start()



while True:
    frame = camera.capture_array()

    frame = frame[:, :, [2,1,0]]
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows() 