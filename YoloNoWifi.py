from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
# flask is a python web framework to build applications easily 
import numpy as np
from pymavlink import mavutil
from gpiozero import AngularServo
from time import sleep, time
import threading 


AUTO_MODE = 0
servo_busy = False
last_trigger_time = 0
TRIGGER_COOLDOWN = 5  


camera = Picamera2()
camera.configure(camera.create_preview_configuration())
camera.start()

#detector for the preliminary target detectiopn (trained on the blank target) 
detector = YOLO('/home/bc/C-UASC/complete_detector_runs/detect/train2/weights/best_ncnn_model')

# classification model to distinguish between the targets
classifier = YOLO('complete_classifier_runs/classify/train/weights/best_ncnn_model')

def zoom_frame(frame, zoom_factor=2.0):
    h, w, _ = frame.shape

    # Compute new cropped size
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)

    # Center crop
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    cropped = frame[y1:y2, x1:x2]

    # Resize back to original size
    zoomed = cv2.resize(cropped, (w, h))

    return zoomed

def mavlink_loop():

    #opens a serial UART connection between the pixhawk and the raspberry pi 
    global master   
    master = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)

    # this bit just confirsms that the connection is active 
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print("Its alive....")

    #requests data from switches @ 10Hz
    master.mav.request_data_stream_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, 10,1
    )
    
    print("looking for switch inputs")
    
    requestedMode = 0
    while True:


        msg = master.recv_match(type='RC_CHANNELS', blocking=True)

        if msg is None:
            continue

        global AUTO_MODE
        remoteControl5 = msg.chan5_raw #SA
        remoteControl6 = msg.chan6_raw #SC

        print(
            f"CH5 = {msg.chan5_raw},"
            f"CH6 = {msg.chan6_raw}"
        )

        #checks to see switch position, numbers represent up, down, middle 

        if remoteControl6 > 1500:
            requestedMode = 3
           
        
        else:
            if remoteControl5 < 1300:
                requestedMode = 0
                
            elif remoteControl5 < 1700:
                requestedMode = 1
               
            else:
                requestedMode = 2
                

        if requestedMode == 0:
            AUTO_MODE = 0
        elif AUTO_MODE == 0:
            AUTO_MODE = requestedMode
        else: 
            pass
        
        if AUTO_MODE == 0:
            print("Manual Mode")
        elif AUTO_MODE == 1:
            print("Target Drop")
        elif AUTO_MODE == 2:
            print("Waypoint Navigation")
        elif AUTO_MODE == 3:
            print("Target Localization")

        print("-----------------------------")
        
            
                

def drop_payload():
    #spins servo to open, then closes, keeps track of whether or not it is busy 
    global servo_busy
    global master
    if not master:
        print("MAVLink connection not established.")
        return  
    print ("dropping payload")
    trigger_servo(master, channel=8, pwm=1900)
    sleep(1)
    trigger_servo(master, channel=8, pwm=1100) 
    servo_busy = False # reset position

def trigger_servo(master, channel=8, pwm=1900):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        channel,
        pwm,
        0, 0, 0, 0, 0
    )            

            

def get_classifier_label(class_results):

    #gets the probabilities and class names from the classifier results
    probs = class_results[0].probs
    names = class_results[0].names

    # Convert NCNN output to normal NumPy array
    #NCNN is a faster format made for running YOLO on chud devices like our PI but it needs to be converted back to a normal format for processing
    prob_data = probs.data

    try:
        prob_data = prob_data.cpu().numpy()
    except Exception:
        prob_data = np.array(prob_data)

    prob_data = np.array(prob_data).reshape(-1)

    class_id = int(np.argmax(prob_data))
    confidence = float(prob_data[class_id])

    class_name = str(names[class_id])

    return class_name, confidence


def detection_loop():
    last_mode = -1
    global servo_busy
    global last_trigger_time

    #prints out current mode, only print if the mode has changed 
    while True:

        if AUTO_MODE != last_mode:
            last_mode = AUTO_MODE

            if AUTO_MODE == 0:
                print("Manual Mode")
            elif AUTO_MODE == 1:
                print("Target Drop")
            elif AUTO_MODE == 2:
                print("Waypoint Navigation")
            elif AUTO_MODE == 3:
                print("Target Localization")

            
        
        # runs the different autonomous routines 
        if AUTO_MODE == 0:
            sleep(0.1)
            continue

        elif AUTO_MODE == 1:

            frame = camera.capture_array()

                #camera settings are wierd, color correct before running detection 

            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = zoom_frame(frame, zoom_factor=2.0)
            frame = np.ascontiguousarray(frame, dtype=np.uint8)

                # Run detector on the first frame that comes in 
            detected_objects = detector(frame, conf=0.4, verbose=False)

            labeled_frame = frame.copy()

            for box in detected_objects[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Gets the bounding boxes coordinates
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                    #this part checks to make sure the bounding box is real

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                    #This runs the calssification yolo model on the CROPPED frame 
                    #returns a confidence value, and alos assings a bounding box, as well as a label
                class_results = classifier(crop, imgsz=224, verbose=False)

                class_name, confidence = get_classifier_label(class_results)

                label = f"{class_name}: {confidence:.2f}"
                global last_trigger_time

                if class_name == "Bullseye" and confidence > 0.8:
                    current_time = time() 
                    
                    if current_time - last_trigger_time > TRIGGER_COOLDOWN:
                        last_trigger_time = current_time
                        if not servo_busy:
                            servo_busy = True
                            threading.Thread(target=drop_payload, daemon=True).start() # reset position

                print(f"[DETECTION] {class_name} ({confidence:.2f}) bbox=({x1},{y1},{x2},{y2})")

            sleep(0.05)

        elif AUTO_MODE == 2:
            
            sleep(0.1)
        elif AUTO_MODE == 3:
           
            sleep(0.1)

#this allows the mavlink to recieve data while also running our autonomous routine so that we can switch to manual if necessary. 
if __name__ == '__main__':
    threading.Thread(target=mavlink_loop, daemon=True).start()
    detection_loop()