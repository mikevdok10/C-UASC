from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
# flask is a python web framework to build applications easily 
import numpy as np
from pymavlink import mavutil
from gpiozero import AngularServo
from time import sleep, time
import threading 
from queue import Queue

AUTO_MODE = 0
servo_busy = False
last_trigger_time = 0
TRIGGER_COOLDOWN = 5  
mavlink_lock = threading.Lock()
 
class Waypoint: 
    def __init__(self, latitude, longitude, altitude):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
 

dronePoisition = {
    "latitude": None, 
    "longitude": None,
    "altitude": None
}


camera = Picamera2()
camera.configure(camera.create_preview_configuration())
camera.start()

# Detector for the preliminary target detection (trained on the blank target) 
detector = YOLO('/home/bc/C-UASC/complete_detector_runs/detect/train2/weights/best_ncnn_model')

# Classification model to distinguish between the targets
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

    # Opens a serial UART connection between the pixhawk and the raspberry pi 
    global master   
    global last_trigger_time
    global servo_busy
    global dronePoisition
    master = mavutil.mavlink_connection('/dev/ttyACM0', baud=57600)


    # Confirms that the connection is active 
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print("Its alive....")

    master.mav.request_data_stream_send(
        master.target_system,
        master.target_component, 
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        10, #10Hz update rate
        1
    )
    
    
    requestedMode = 0

    while True:
        try:
            with mavlink_lock:
                msg = master.recv_match(blocking=False)
            if msg is None:
                continue
            msg_type = msg.get_type()

        except Exception as e:
            print(f"MAVLink error: {e}, reconnecting...")
            sleep(2)
            with mavlink_lock:
                master = mavutil.mavlink_connection('/dev/ttyACM0', baud=57600)
                master.wait_heartbeat()
            print("Reconnected")

        if msg_type == "GLOBAL_POSITION_INT":
            with mavlink_lock:
                dronePoisition["latitude"] = msg.lat / 1e7
                dronePoisition["longitude"] = msg.lon / 1e7
                dronePoisition["altitude"] = msg.alt / 1000.0
            print(
            f"Current Position: lat={dronePoisition['latitude']}," 
            f"  lon={dronePoisition['longitude']}, "
            f"alt={dronePoisition['altitude']}"
            )
            print("------------------------------------")
        if msg_type == "HEARTBEAT":
            print(mavutil.mode_string_v10(msg))
        

        elif msg_type == "RC_CHANNELS":
            with mavlink_lock:
                global AUTO_MODE
                remoteControl5 = msg.chan5_raw #SA
                remoteControl6 = msg.chan6_raw #SC
                flgihtModeSwitch = msg.chan8_raw #SB
                manualDrop = msg.chan7_raw #SD
                print(
                    f"CH5 = {msg.chan5_raw},"
                    f"CH6 = {msg.chan6_raw},"
                    f"CH7 = {msg.chan7_raw}"
                    f"CH8 = {msg.chan8_raw}"
                )


                if manualDrop > 1800:
                    current_time = time()

                    if current_time - last_trigger_time > TRIGGER_COOLDOWN:
                        last_trigger_time = current_time

                        if not servo_busy:
                            servo_busy = True
                            threading.Thread(target=drop_payload, daemon=True).start()
                if flgihtModeSwitch < 1230:
                    master.mav.command_long_send(
                    
                        master.target_system,
                        master.target_component,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        5, 0, 0, 0, 0, 0, 0, 0
                    )
                elif flgihtModeSwitch > 1230:
                    master.mav.command_long_send(
                        master.target_system,
                        master.target_component,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        1, 0, 0, 0, 0, 0, 0, 0
                    )

                    if remoteControl6 > 1900:
                        requestedMode = 4 #Waypoint Navigation 
                    elif remoteControl6 > 1400:
                        requestedMode = 3 # Target Localization
                    
                    else:
                        if remoteControl5 < 1300:
                            requestedMode = 0 #Manual Mode
                            
                        elif remoteControl5 < 1700:
                            requestedMode = 1 #Target Drop
                        
                        else:
                            requestedMode = 2 #Waypoint Navigation
                            
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
                        print("Package Delivery")
                    elif AUTO_MODE == 3:
                        print("Target Localization")
                    elif AUTO_MODE == 4:
                        print("Waypoint Navigation")

                    print("************************************")
                    
                else:
                    master.mav.command_long_send(
                        master.target_system,
                        master.target_component,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        6, 0, 0, 0, 0, 0, 0, 0
                    )
                #checks to see switch position, numbers represent up, down, middle 



def drop_payload():
    # Spins servo to open, then closes, keeps track of whether or not it is busy 
    global servo_busy

    if not master:
        print("MAVLink connection not established.")
        return  
    
    print ("dropping payload")
    trigger_servo(master, channel=9, pwm=1200)
    sleep(1)

    print("closing servo")
    trigger_servo(master, channel=9, pwm=600) 
    servo_busy = False # reset position

def trigger_servo(master, channel=9, pwm=1400):
    with mavlink_lock:
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

    # Gets the probabilities and class names from the classifier results
    probs = class_results[0].probs
    names = class_results[0].names

    # Convert NCNN output to normal NumPy array
    # NCNN is a faster format made for running YOLO on chud devices like our Pi but it needs to be converted back to a normal format for processing
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

    # Prints out current mode, only print if the mode has changed 
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

            
        # ---------< AUTONOMOUS ROUTINES >----------
        
        # 0 - Manual Mode 
        if AUTO_MODE == 0: 
            sleep(0.1)
            continue

        # 1 - Target Drop   
        elif AUTO_MODE == 1:

            frame = camera.capture_array()

            # Color correction
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

                # Checks to make sure the bounding box is real
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # This runs the classification yolo model on the CROPPED frame 
                # Returns a confidence value, and alos assings a bounding box, as well as a label
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

        # 2 - Package Delivery 
        elif AUTO_MODE == 2: 

            frame = camera.capture_array()

            # Color correction (again)
            # Lots of copy pasted code from the previous mode, needs to be cleaned up
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

                # This part checks to make sure the bounding box is real
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # This runs the classification yolo model on the CROPPED frame 
                # Returns a confidence value, and alos assings a bounding box, as well as a label
                class_results = classifier(crop, imgsz=224, verbose=False)

                class_name, confidence = get_classifier_label(class_results)

                label = f"{class_name}: {confidence:.2f}"
                

                if class_name == "Bullseye" and confidence > 0.8:
                    current_time = time() 
                    
                    if current_time - last_trigger_time > TRIGGER_COOLDOWN:
                        last_trigger_time = current_time
                        if not servo_busy:
                            servo_busy = True
                            threading.Thread(target=drop_payload, daemon=True).start() # reset position

                    print(f"[DETECTION] {class_name} ({confidence:.2f}) bbox=({x1},{y1},{x2},{y2})")
            
            sleep(0.05)

        # 3 - Target Localization
        elif AUTO_MODE == 3: 
           
            sleep(0.1)

        # 4 - Waypoint Navigation
        elif AUTO_MODE == 4: 
            
            sleep(0.1)

# Allows the mavlink to recieve data while also running  autonomous routine so that we can switch to manual if necessary
if __name__ == '__main__':
    threading.Thread(target=mavlink_loop, daemon=True).start()
    detection_loop()