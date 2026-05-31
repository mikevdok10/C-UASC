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
import math

AUTO_MODE = 0

AUTO_MODE_NAMES = {
    0: "Manual",
    1: "Target Drop",
    2: "Package Delivery",
    3: "Target Localization",
    4: "Waypoint Navigation"
}

COPTER_MODES = {
    "Stabalize":0,
    "ALTHOLD":2,
    "AUTO":3,
    "GUIDED":4,
    "LOITER":5,
    "RTL":6,
}
servo_busy = False
last_trigger_time = 0
TRIGGER_COOLDOWN = 5  
mavlink_lock = threading.Lock()
current_requested_pixhawk_mode = None
 
SEARCH_ALTITUDE = 10.0
class Waypoint: 
    def __init__(self, latitude, longitude, altitude):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
 
PACKAGE_DELIVERY_WAYPOINTS = [
    Waypoint(37.7749, -122.4194, SEARCH_ALTITUDE),  # Example waypoint 1
    Waypoint(37.7750, -122.4180, SEARCH_ALTITUDE),  # Example waypoint 2
    Waypoint(37.7755, -122.4170, SEARCH_ALTITUDE)   # Example waypoint 3
]

NAV_WAYPOINTS = [
    Waypoint(37.7749, -122.4194, SEARCH_ALTITUDE),  # Example waypoint 1
    Waypoint(37.7750, -122.4180, SEARCH_ALTITUDE),  # Example waypoint 2
    Waypoint(37.7755, -122.4170, SEARCH_ALTITUDE)   # Example waypoint 3
]



dronePoisition = {

    
    "latitude": None, 
    "longitude": None,
    "altitude": None
}

localized_targets = []


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
def has_position():
    return (
        dronePoisition["latitude"] is not None and
        dronePoisition["longitude"] is not None and
        dronePoisition["altitude"] is not None
    )

def distance_meters(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two GPS coordinates using Haversine formula."""
    R = 6371000  # Earth radius in meters
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def set_flight_mode(mode):
    global master

    if master is None:
        print("MAVLink connection not established.")
        return False
    
    if mode not in COPTER_MODES:
        print(f"Unknown flight mode: {mode}")
        return False

    mode_id = COPTER_MODES[mode]

    with mavlink_lock:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mode_id,
            0, 0, 0, 0, 0, 0
        )
    print(f"[MODE] Requested mode change to {mode} (ID: {mode_id})")
    return True

def request_pixhawk_mode(mode):
    """Requests a flight mode change on the Pixhawk."""
    global current_requested_pixhawk_mode

    mode = mode.upper()

    if mode != current_requested_pixhawk_mode:
        success = set_flight_mode(mode)
        if success:
            current_requested_pixhawk_mode = mode

def request_auto_mode(requested_mode):
    """Controls the AUTO_MODE state machine and requests Pixhawk mode changes as needed."""
    global AUTO_MODE
    if requested_mode == 0:  # Manual Mode
        if AUTO_MODE != 0:
            print("[AUTO_MODE] Switching to Manual Mode.")
        AUTO_MODE = 0
        return

    if AUTO_MODE == 0:
        AUTO_MODE = requested_mode
        print(f"[AUTO_MODE] Switching to {AUTO_MODE_NAMES.get(AUTO_MODE, 'Unknown')} Mode.")
    else:
        print(f"[AUTO_MODE] Already in {AUTO_MODE_NAMES.get(AUTO_MODE, 'Unknown')} Mode. Ignoring request to switch to {AUTO_MODE_NAMES.get(requested_mode, 'Unknown')} Mode.")

def goto_location(lat, lon, alt):
    if master is None:
        print("MAVLink connection not established.")
        return False

    with mavlink_lock:
        master.mav.set_position_target_global_int_send(
            0,
            master.target_system,
            master.target_component,  # seq
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                0b110111111000,  # type_mask (only position)
                int(lat * 1e7),  # latitude
                int(lon * 1e7),  # longitude
                alt,  # altitude in mm
                0, 0, 0,  # velocity (not used)
                0, 0, 0,  # acceleration (not used)
                0, 0  # yaw, yaw_rate (not used)
        )
    print(f"[GUIDED] Sent waypoint command to lat={lat}, lon={lon}, alt={alt}")
    
def send_local_velocity(vx, vy, vz, duration):
    """Sends a velocity command in the local frame for a specified duration. vx=forward/back, vy=left/right, vz=up/down."""
    if master is None:
        print("MAVLink connection not established.")
        return False

    end_time = time() + duration
    while time() < end_time:
        with mavlink_lock:
            master.mav.set_position_target_local_ned_send(
                0,
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b1100111000111,  # type_mask (only velocity)
                0, 0, 0,  # x, y, z positions (not used)
                vx, vy, vz,  # velocities in m/s
                0, 0, 0,  # accelerations (not used)
                0, 0  # yaw, yaw_rate (not used)
            )
        sleep(0.2)
        
def wait_until_reached(target_lat, target_lon, radius = 2.0, timeout = 30):
    """Waits until the drone reaches a specified location."""
    start = time()
    while time() - start < timeout:
        if AUTO_MODE == 0:
            print("[WAYPOINT] Cancelled by manual mode.")
            return False
        
        lat = dronePoisition["latitude"]
        lon = dronePoisition["longitude"]

        if lat is None or lon is None:
            sleep(0.2)
            continue
        distance = distance_meters(lat, lon, target_lat, target_lon)
        print(f"[WAYPOINT] Distance to target: {distance:.2f} meters")
        if distance <= radius:
            print("[WAYPOINT] Target reached.")
            return True
        sleep(0.5)
    print("[WAYPOINT] Timeout reached without arriving at target.")
    return False

# ============================================================
# PAYLOAD / SERVO HELPERS
# ============================================================
def trigger_servo(channel=9, pwm=1400):
    if master is None:
        print("[SERVO]MAVLink connection not established.")
        return  
    
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

def drop_payload():
    """spins servo to open, then closes, keeps track of whether or not it is busy"""
    global servo_busy
    if master is None:
        print("[DROP] MAVLink connection not established.")
        servo_busy = False
        return  
    
    print ("[DROP] Dropping payload...")
    trigger_servo(channel=9, pwm=1200)
      # Open position
    sleep(1)

    print("[DROP] Closing servo...")
    trigger_servo(channel=9, pwm=600)   # Closed position
    servo_busy = False  # Reset busy state
  # Send command at 10Hz

def try_drop_payload():
    """starts the payload drop thread if not already busy and cooldown has passed"""
    global servo_busy
    global last_trigger_time

    current_time = time() 
                    
    if current_time - last_trigger_time <= TRIGGER_COOLDOWN:
        return False
    if servo_busy:
        return False
    last_trigger_time = current_time
    servo_busy = True

    threading.Thread(target=drop_payload, daemon=True).start()
    return True
        
# ============================================================
# VISION / YOLO HELPERS
# ============================================================

def get_classifier_label(class_results):
    """Extracts the class name and confidence from the classifier results."""
    probs = class_results[0].probs
    names = class_results[0].names

    # Convert NCNN output to normal NumPy array
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

def detect_target():
    """Captures a frame, runs detection and classification, and returns results."""

    frame = camera.capture_array()

    # Camera settings are weird, color correct before running detection 
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = zoom_frame(frame, zoom_factor=2.0)
    frame = np.ascontiguousarray(frame, dtype=np.uint8)

    # Run detector on the first frame that comes in 
    detected_objects = detector(frame, conf=0.4, verbose=False)

    best_detection = None
    best_confidence = 0.0
    for box in detected_objects[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Gets the bounding boxes coordinates
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        class_results = classifier(crop, imgsz=224, verbose=False)

        class_name, confidence = get_classifier_label(class_results)

        if confidence > best_confidence:
            best_confidence = confidence
            best_detection = {
                "class_name": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            }
        
    return best_detection

# ============================================================
# AUTONOMOUS ROUTINES
# ============================================================

def routine_target_drop():
    """Autonomous routine for target drop mode."""
    global AUTO_MODE
    
    print("[ROUTINE] Starting Target Drop routine.")
    request_pixhawk_mode("GUIDED")

    while AUTO_MODE == 1:
        detection = detect_target()
        if detection is None:
            sleep(0.05)
            continue
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        print(f"[TARGET DROP] Detected {class_name} with confidence {confidence:.2f}")

        if class_name == "Bullseye" and confidence > 0.8:
            if try_drop_payload():
                print(f"[TARGET DROP] Payload drop triggered for {class_name} (confidence: {confidence:.2f})")

                AUTO_MODE = 0
                request_pixhawk_mode("STABILIZE")
                """check with Peter if this the correct arming mode, and if we need to disarm after each drop or not. Also check if we need to switch back to stabilize after each drop or not, or if we can just stay in guided and keep sending the drop command when we see a target. """
                return
        sleep(0.05)

    print("[ROUTINE] Exiting Target Drop routine.")

def routine_package_delivery():
    """AutoMode 2: Package Delivery routine - fly to predefined waypoints and drop payload."""
    global AUTO_MODE
    
    print("[ROUTINE] Starting Package Delivery routine.")
    if not has_position():
        print("[PACKAGE DELIVERY] Current position unknown, cannot start package delivery.")
        AUTO_MODE = 0
        return
    request_pixhawk_mode("GUIDED")

    for waypoint in PACKAGE_DELIVERY_WAYPOINTS:
        if AUTO_MODE != 2:
            print("[ROUTINE] Package Delivery routine cancelled.")
            return
        
        print(f"[ROUTINE] Navigating to waypoint at lat={waypoint.latitude}, lon={waypoint.longitude}, alt={waypoint.altitude}")
        goto_location(waypoint.latitude, waypoint.longitude, waypoint.altitude)
        reached = False
        waypoint_start = time()

        while AUTO_MODE == 2 and not reached:
            detection = detect_target()

            if detection:
                class_name = detection["class_name"]
                confidence = detection["confidence"]

                print(f"[PACKAGE DELIVERY] Detected {class_name} with confidence {confidence:.2f} while en route to waypoint.")

                if class_name == "Bullseye" and confidence > 0.8:
                    print("[PACKAGE DELIVERY] Target detected near waypoint, attempting payload drop...")
                    if try_drop_payload():
                        print("[PACKAGE DELIVERY] Payload drop triggered by target detection.")
                    
                    lat = dronePosition["latitude"]
                    lon = dronePosition["longitude"]

                    if lat is not None and lon is not None:
                        goto_location(lat, lon, SEARCH_ALTITUDE)
                        sleep(1)
                    
                    if try_drop_payload():
                        print("[PACKAGE DELIVERY] Payload drop triggered at current location.")

                    AUTO_MODE = 0
                    request_pixhawk_mode("STABILIZE")
                    return
            lat = dronePosition["latitude"]
            lon = dronePosition["longitude"]
            if lat is not None and lon is not None:
                dist = distance_meters(lat, lon, waypoint.latitude, waypoint.longitude)
                if dist < 2.0:
                    reached = True
                    print("[PACKAGE DELIVERY] Waypoint reached.")
            
            if time() - waypoint_start > 40:
                print("[PACKAGE DELIVERY] Timeout reached while trying to reach waypoint.")
                reached = True
                sleep(0.1)

    print("[ROUTINE] Package Delivery routine completed all waypoints.")
    AUTO_MODE = 0
    request_pixhawk_mode("STABILIZE")

def routine_target_localization():
    """
    AUTO_MODE 3:
    Searches visually for Bullseye.
    When seen, records current drone GPS position.

    NOTE:
    This does not calculate the exact ground coordinate of the target.
    It records where the drone was when the target was seen.
    True target geolocation requires altitude AGL, attitude, camera calibration,
    camera mounting angle, and pixel offset from image center.
    """

    global AUTO_MODE

    print("[ROUTINE] Target Localization started.")

    if not has_position():
        print("[LOCALIZE] No GPS position yet. Cannot localize target.")
        AUTO_MODE = 0
        return

    request_pixhawk_mode("GUIDED")

    while AUTO_MODE == 3:
        detection = detect_target()

        if detection is None:
            sleep(0.05)
            continue

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        print(f"[LOCALIZE] Saw {class_name} confidence={confidence:.2f}")

        if class_name == "Bullseye" and confidence > 0.80:
            lat = dronePosition["latitude"]
            lon = dronePosition["longitude"]
            alt = dronePosition["altitude"]

            if lat is not None and lon is not None:
                target_data = {
                    "class_name": class_name,
                    "confidence": confidence,
                    "drone_latitude": lat,
                    "drone_longitude": lon,
                    "drone_altitude": alt,
                    "time": time(),
                }

                localized_targets.append(target_data)

                print("[LOCALIZE] Target recorded:")
                print(target_data)

                AUTO_MODE = 0
                request_pixhawk_mode("LOITER")
                return

        sleep(0.05)

    print("[ROUTINE] Target Localization exited.")

def routine_waypoint_navigation():
    """
    AUTO_MODE 4:
    Flies through a list of GPS waypoints.
    """

    global AUTO_MODE

    print("[ROUTINE] Waypoint Navigation started.")

    if not has_position():
        print("[WAYPOINT] No GPS position yet. Cannot start waypoint navigation.")
        AUTO_MODE = 0
        return

    request_pixhawk_mode("GUIDED")

    for wp in NAV_WAYPOINTS:
        if AUTO_MODE != 4:
            print("[WAYPOINT] Cancelled.")
            return

        goto_location(wp.latitude, wp.longitude, wp.altitude)

        reached = wait_until_reached(
            wp.latitude,
            wp.longitude,
            radius=2.0,
            timeout=45,
        )

        if not reached:
            print("[WAYPOINT] Failed to reach waypoint. Moving on.")

    print("[WAYPOINT] Route complete. Switching to LOITER.")
    AUTO_MODE = 0
    request_pixhawk_mode("LOITER")


def autonomy_loop():
    """
    Runs the selected autonomous routine.
    Each routine exits if AUTO_MODE changes.
    """

    last_printed_mode = None

    while True:
        mode = AUTO_MODE

        if mode != last_printed_mode:
            print(f"[AUTO MODE] {AUTO_MODE_NAMES.get(mode, 'Unknown')}")
            last_printed_mode = mode

        if mode == 0:
            sleep(0.1)
            continue

        if mode == 1:
            routine_target_drop()

        elif mode == 2:
            routine_package_delivery()

        elif mode == 3:
            routine_target_localization()

        elif mode == 4:
            routine_waypoint_navigation()

        else:
            print(f"[AUTONOMY] Unknown AUTO_MODE: {mode}")
            sleep(0.1)


# ============================================================
# MAVLINK LOOP / RC SWITCHING LOGIC
# ============================================================

def handle_rc_channels(msg):
    """
    Reads RC channels and updates the custom AUTO_MODE.

    Current switch plan:

    CH8 / SB:
        low    = RTL emergency/recovery
        middle = AltHold pilot-controlled
        high   = Guided companion-computer control allowed

    CH5 / SA:
        low    = Manual/no autonomous routine
        middle = Target Drop
        high   = Package Delivery

    CH6 / SC:
        middle = Target Localization
        high   = Waypoint Navigation

    CH7 / SD:
        high   = Manual payload drop
    """

    remote_control_5 = msg.chan5_raw  # SA
    remote_control_6 = msg.chan6_raw  # SC
    manual_drop = msg.chan7_raw       # SD
    flight_mode_switch = msg.chan8_raw  # SB

    print(
        f"CH5={remote_control_5}, "
        f"CH6={remote_control_6}, "
        f"CH7={manual_drop}, "
        f"CH8={flight_mode_switch}"
    )

    # Manual payload drop
    if manual_drop > 1800:
        if try_drop_payload():
            print("[MANUAL DROP] Payload drop triggered.")

    # SB low: emergency/recovery RTL
    if flight_mode_switch < 1300:
        request_auto_mode(0)
        request_pixhawk_mode("RTL")
        return

    # SB middle: pilot controlled AltHold
    if flight_mode_switch < 1700:
        request_auto_mode(0)
        request_pixhawk_mode("ALTHOLD")
        return

    # SB high: companion computer allowed
    request_pixhawk_mode("GUIDED")

    # SC has priority over SA for specialized autonomous modes
    if remote_control_6 > 1900:
        requested_mode = 4  # Waypoint Navigation
    elif remote_control_6 > 1400:
        requested_mode = 3  # Target Localization
    else:
        if remote_control_5 < 1300:
            requested_mode = 0  # Manual
        elif remote_control_5 < 1700:
            requested_mode = 1  # Target Drop
        else:
            requested_mode = 2  # Package Delivery

    request_auto_mode(requested_mode)


def mavlink_loop():
    global master
    global dronePosition

    # Opens a serial USB connection between the Pixhawk and Raspberry Pi
    master = mavutil.mavlink_connection("/dev/ttyACM0", baud=57600)

    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print("Heartbeat received. Pixhawk is alive.")

    # Request data streams from Pixhawk
    master.mav.request_data_stream_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        10,  # 10Hz update rate
        1,
    )

    while True:
        try:
            with mavlink_lock:
                msg = master.recv_match(blocking=False)

            if msg is None:
                sleep(0.01)
                continue

            msg_type = msg.get_type()

        except Exception as e:
            print(f"[MAVLINK] Error: {e}. Reconnecting...")
            sleep(2)

            with mavlink_lock:
                master = mavutil.mavlink_connection("/dev/ttyACM0", baud=57600)
                master.wait_heartbeat()

            print("[MAVLINK] Reconnected.")
            continue

        if msg_type == "GLOBAL_POSITION_INT":
            dronePosition["latitude"] = msg.lat / 1e7
            dronePosition["longitude"] = msg.lon / 1e7
            dronePosition["altitude"] = msg.relative_alt / 1000.0

            print(
                f"Current Position: "
                f"lat={dronePosition['latitude']}, "
                f"lon={dronePosition['longitude']}, "
                f"alt={dronePosition['altitude']}"
            )
            print("------------------------------------")

        elif msg_type == "HEARTBEAT":
            print(f"[HEARTBEAT] Current Pixhawk mode: {mavutil.mode_string_v10(msg)}")

        elif msg_type == "RC_CHANNELS":
            handle_rc_channels(msg)
            print("************************************")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    threading.Thread(target=mavlink_loop, daemon=True).start()
    autonomy_loop()

