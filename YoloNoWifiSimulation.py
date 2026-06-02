from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import numpy as np
from pymavlink import mavutil
from gpiozero import AngularServo
from time import sleep, time
import threading 
from queue import Queue
import math

last_rc_print_time = 0
last_position_print_time = 0
last_heartbeat_print_time = 0
SERIAL_PRINT_INTERVAL = 1.0

STREAM_HOST = "172.20.10.10"
Stream_Port = 5600
Stream_Width = 640
Stream_Height = 480
Stream_FPS = 30
Stream_Bitrate = 1000
Stream_enabled = True 

latest_frame = None
latest_frame_time = 0

camera_lock = threading.Lock()


AUTO_MODE = 0

KEYBOARD_TEST_MODE = True
auto_mode_lock = threading.Lock()

master = None

AUTO_MODE_NAMES = {
    0: "Manual",
    1: "Target Drop",
    2: "Package Delivery",
    3: "Target Localization",
    4: "Waypoint Navigation"
}

COPTER_MODES = {
    "GUIDED":4,
    "LOITER":5,
    "RTL":6,
}
servo_busy = False
last_trigger_time = 0
TRIGGER_COOLDOWN = 5  
mavlink_lock = threading.Lock()
current_requested_pixhawk_mode = None


 
SEARCH_ALTITUDE = 8.5

def set_mode(mode_name):
    global master

    if master is None:
        print(f"[MODE] Cannot set {mode_name}. MAVLink is not connected yet.")
        return False

    try:
        modes = master.mode_mapping()

        if mode_name not in modes:
            print(f"[MODE] Mode {mode_name} not available. Available modes: {list(modes.keys())}")
            return False

        mode_id = modes[mode_name]

        with mavlink_lock:
            master.mav.set_mode_send(
                master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )

        print(f"[MODE] Mode set command sent: {mode_name}")
        return True

    except Exception as e:
        print(f"[MODE] Failed to set mode {mode_name}: {e}")
        return False

class Waypoint: 
    def __init__(self, latitude, longitude, altitude):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
 
def send_body_velocity(forward_m_s, right_m_s, down_m_s):
    """
    BODY_NED frame:
    +X = forward
    +Y = right
    +Z = down
    So up is negative Z.
    """

    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,  # only velocity enabled
        0, 0, 0,             # position ignored
        forward_m_s,
        right_m_s,
        down_m_s,
        0, 0, 0,             # acceleration ignored
        0, 0                 # yaw, yaw_rate ignored
    )


def move_forward_for_seconds(speed_m_s, seconds):
    print(f"Moving forward at {speed_m_s} m/s for {seconds} seconds")

    start_time = time.time()

    while time.time() - start_time < seconds:
        send_body_velocity(speed_m_s, 0, 0)
        time.sleep(0.1)  # 10 Hz

    send_body_velocity(0, 0, 0)
    print("Stopped")


def goto_coordinate(latitude, longitude, altitude, seconds=5):
    """
    Moves the drone to a GPS coordinate using SET_POSITION_TARGET_GLOBAL_INT.
    Assumes the drone is already in GUIDED mode, armed, and flying.
    """

    lat_int = int(latitude * 1e7)
    lon_int = int(longitude * 1e7)

    # Position enabled, velocity/acceleration/yaw ignored.
    type_mask = 0b0000111111111000

    print(f"Going to: lat={latitude}, lon={longitude}, alt={altitude}")

    start_time = time()

    while time() - start_time < seconds:
        master.mav.set_position_target_global_int_send(
            0,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            type_mask,
            lat_int,
            lon_int,
            altitude,
            0, 0, 0,     # velocity ignored
            0, 0, 0,     # acceleration ignored
            0, 0         # yaw, yaw_rate ignored
        )

        time.sleep(0.2)


def lawnmowerPath(coordinatePoints, spacingBetweenPaths):
    """
    Creates a local x/y lawnmower path inside the bounding box of the given local points.
    coordinatePoints should be a list of (x, y) tuples in meters.
    """

    if coordinatePoints is None:
        raise ValueError("coordinatePoints is None. Pass in a list of local (x, y) points.")

    if len(coordinatePoints) == 0:
        raise ValueError("coordinatePoints is empty.")

    if spacingBetweenPaths <= 0:
        raise ValueError("spacingBetweenPaths must be greater than 0.")

    xs = [p[0] for p in coordinatePoints]
    ys = [p[1] for p in coordinatePoints]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    path = []

    y = min_y
    direction = 1

    while y <= max_y:
        if direction == 1:
            path.append((min_x, y))
            path.append((max_x, y))
        else:
            path.append((max_x, y))
            path.append((min_x, y))

        y += spacingBetweenPaths
        direction *= -1  # alternate direction each row

    # Make sure the top edge gets covered even if spacing does not land exactly on max_y.
    if path and path[-1][1] < max_y:
        if direction == 1:
            path.append((min_x, max_y))
            path.append((max_x, max_y))
        else:
            path.append((max_x, max_y))
            path.append((min_x, max_y))

    return path


def lawnmowerToGPS(localPath, ref_lat, ref_lon, altitude):
    if localPath is None:
        raise ValueError("localPath is None. lawnmowerPath() probably did not return a path.")

    gps_path = []

    for x, y in localPath:
        lat, lon = local_to_gps(x, y, ref_lat, ref_lon)
        gps_path.append((lat, lon, altitude))

    return gps_path


def lawnmowerSearch(localPoints, ref_lat, ref_lon, altitude, spacingBetweenPaths):
    local_path = lawnmowerPath(localPoints, spacingBetweenPaths)
    gps_path = lawnmowerToGPS(local_path, ref_lat, ref_lon, altitude)

    print("Lawnmower GPS Path:")
    for lat, lon, alt in gps_path:
        print(lat, lon, alt)

    for lat, lon, alt in gps_path:
        goto_coordinate(lat, lon, alt, seconds=5)
        time.sleep(1)

    return gps_path


class Waypoint:
    def __init__(self, latitude, longitude, altitude):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude


radiusEarth = 6378137.0


def gps_to_local(lat, lon, ref_lat, ref_lon):
    x = math.radians(lon - ref_lon) * radiusEarth * math.cos(math.radians(ref_lat))
    y = math.radians(lat - ref_lat) * radiusEarth
    return x, y


def local_to_gps(x, y, ref_lat, ref_lon):
    lat = ref_lat + math.degrees(y / radiusEarth)
    lon = ref_lon + math.degrees(x / (radiusEarth * math.cos(math.radians(ref_lat))))
    return lat, lon


# GPS bounding box corners
gpsCoordinate_1 = Waypoint(-35.3622723, 149.1657758, 2)
gpsCoordinate_2 = Waypoint(-35.3623292, 149.1648316, 2)
gpsCoordinate_3 = Waypoint(-35.3635847, 149.1650730, 2)
gpsCoordinate_4 = Waypoint(-35.3634185, 149.1660118, 2)

targetDropTestCoordinate = Waypoint(35.4060143, -118.970300, 2)

boundingBoxCorners = [
    gpsCoordinate_1,
    gpsCoordinate_2,
    gpsCoordinate_3,
    gpsCoordinate_4
]

ref_lat = boundingBoxCorners[0].latitude
ref_lon = boundingBoxCorners[0].longitude

local_pts = [
    gps_to_local(
        wp.latitude,
        wp.longitude,
        ref_lat,
        ref_lon
    )
    for wp in boundingBoxCorners
]


dronePosition = {

    
    "latitude": None, 
    "longitude": None,
    "altitude": None
}

localized_targets = []


camera = Picamera2()
camera.configure(camera.create_video_configuration(
    main ={"size": (Stream_Width, Stream_Height), "format": "RGB888"}, 
    controls={"FrameRate": Stream_FPS}
))
camera.start()

#detector for the preliminary target detectiopn (trained on the blank target) 
detector = YOLO('/home/bc/C-UASC/complete_detector_runs/detect/train2/weights/best_ncnn_model')

# classification model to distinguish between the targets
classifier = YOLO('complete_classifier_runs/classify/train/weights/best_ncnn_model')


def camera_loop():
    global latest_frame
    global latest_frame_time

    while True:
        try:
            frame = camera.capture_array()

            if frame is None:
                sleep(0.01)
                continue

            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            with camera_lock:
                latest_frame = frame.copy()
                latest_frame_time = time()

        except Exception as e:
            print(f"[CAMERA] Error capturing frame: {e}")
            sleep(0.1)

def get_latest_frame(timeout=2.0):
    start = time()
    while time() - start < timeout:
        with camera_lock:
            if latest_frame is not None:
                return latest_frame.copy()
        sleep(0.05)
    
    print("[CAMERA] Timeout waiting for latest frame.")
    return None 
def build_gstreamer_pipeline():
    return (
        f"appsrc is-live=true block=true format=time "
        f"caps=video/x-raw,format=BGR,width={Stream_Width},height={Stream_Height},framerate={Stream_FPS}/1 ! "
        "videoconvert ! "
        "video/x-raw,format=I420 ! "
        f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={Stream_Bitrate} key-int-max={Stream_FPS} ! "
        "h264parse config-interval=1 ! "
        "mpegtsmux ! "
        f"udpsink host={STREAM_HOST} port={Stream_Port} sync=false async=false"
    )

def gstreamer_loop():
    if not Stream_enabled:
        print("[STREAM] Streaming disabled.")
        return

    pipeline = build_gstreamer_pipeline()

    print(f"[STREAM] Starting UDP stream to udp://{STREAM_HOST}:{Stream_Port}")
    print(f"[STREAM] Pipeline: {pipeline}")

    writer = cv2.VideoWriter(
        pipeline,
        cv2.CAP_GSTREAMER,
        0,
        Stream_FPS,
        (Stream_Width, Stream_Height),
        True,
    )

    if not writer.isOpened():
        print("[STREAM] Failed to open GStreamer VideoWriter.")
        return

    frame_interval = 1.0 / Stream_FPS
    next_frame_time = time()

    while True:
        frame_rgb = get_latest_frame(timeout=2.0)

        if frame_rgb is None:
            sleep(0.1)
            continue

        if frame_rgb.shape[1] != Stream_Width or frame_rgb.shape[0] != Stream_Height:
            frame_rgb = cv2.resize(frame_rgb, (Stream_Width, Stream_Height))

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        sleep_time = next_frame_time - time()
        if sleep_time > 0:
            sleep(sleep_time)

        next_frame_time += frame_interval


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
        sleep(0.1)


def force_auto_mode(requested_mode):
    """
    Keyboard/testing version of request_auto_mode().
    This allows you to swap between autonomous modes immediately.
    Useful for bench testing without RC switches.
    """
    global AUTO_MODE

    if requested_mode not in AUTO_MODE_NAMES:
        print(f"[KEYBOARD] Invalid AUTO_MODE: {requested_mode}")
        return

    with auto_mode_lock:
        if AUTO_MODE != requested_mode:
            print(
                f"[KEYBOARD] Switching from "
                f"{AUTO_MODE_NAMES.get(AUTO_MODE, 'Unknown')} "
                f"to {AUTO_MODE_NAMES.get(requested_mode, 'Unknown')}"
            )

        AUTO_MODE = requested_mode


def keyboard_test_loop():
    """
    Lets you switch autonomous modes using keyboard keys.

    Keys:
        0 = Manual / stop autonomous routine
        1 = Target Drop
        2 = Package Delivery
        3 = Target Localization
        4 = Waypoint Navigation

        l = LOITER
        g = GUIDED
        r = RTL

        q = quit keyboard loop
    """

    import sys
    import termios
    import tty
    import select

    print("[KEYBOARD] Keyboard test mode active.")
    print("[KEYBOARD] Press 0-4 to change AUTO_MODE.")
    print("[KEYBOARD] Press l=LOITER, g=GUIDED, r=RTL, q=quit keyboard thread.")

    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1).lower()

                if key == "0":
                    force_auto_mode(0)
                    set_mode("LOITER")

                elif key == "1":
                    force_auto_mode(1)
                    set_mode("GUIDED")

                elif key == "2":
                    force_auto_mode(2)
                    set_mode("GUIDED")

                elif key == "3":
                    force_auto_mode(3)
                    set_mode("GUIDED")

                elif key == "4":
                    force_auto_mode(4)
                    set_mode("GUIDED")

                elif key == "l":
                    force_auto_mode(0)
                    set_mode("LOITER")

                elif key == "g":
                    set_mode("GUIDED")

                elif key == "r":
                    force_auto_mode(0)
                    set_mode("RTL")

                elif key == "q":
                    print("[KEYBOARD] Exiting keyboard test loop.")
                    break

                else:
                    print(f"[KEYBOARD] Unknown key: {key}")

            sleep(0.05)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
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

    frame_rgb = get_latest_frame(timeout=2.0)
    if frame_rgb is None:
        sleep(0.05)
        return None
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

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

        print(f"[VISION] Classifier says: {class_name}, confidence={confidence:.2f}")


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
    """AUTO_MODE 1: fly toward target coordinate while constantly running detection."""
    global AUTO_MODE

    print("[ROUTINE] Starting Target Drop routine.")
    set_mode("GUIDED")

    last_goto_send_time = 0
    last_no_detection_print_time = 0

    # Convert once instead of every loop
    lat_int = int(targetDropTestCoordinate.latitude * 1e7)
    lon_int = int(targetDropTestCoordinate.longitude * 1e7)
    altitude = targetDropTestCoordinate.altitude

    # Position enabled, velocity/acceleration/yaw ignored
    type_mask = 0b0000111111111000

    while AUTO_MODE == 1:
        current_time = time()

        # Keep sending the goto command while also allowing detection to run.
        # 5 Hz is enough for GUIDED position target updates.
        if current_time - last_goto_send_time >= 0.2:
            with mavlink_lock:
                master.mav.set_position_target_global_int_send(
                    0,
                    master.target_system,
                    master.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                    type_mask,
                    lat_int,
                    lon_int,
                    altitude,
                    0, 0, 0,     # velocity ignored
                    0, 0, 0,     # acceleration ignored
                    0, 0         # yaw, yaw_rate ignored
                )

            last_goto_send_time = current_time

        # Detection now runs the whole time AUTO_MODE == 1
        detection = detect_target()

        if detection is None:
            if current_time - last_no_detection_print_time >= 1.0:
                print("[TARGET DROP] Searching... no target detected.")
                last_no_detection_print_time = current_time

            sleep(0.05)
            continue

        class_name = detection["class_name"]
        confidence = detection["confidence"]

        print(f"[TARGET DROP] Detected {class_name} with confidence {confidence:.2f}")

        if "bullseye" in class_name.lower() and confidence > 0.8:
            if try_drop_payload():
                print(f"[TARGET DROP] Payload drop triggered for {class_name} confidence={confidence:.2f}")

                AUTO_MODE = 0
                set_mode("LOITER")
                return

        sleep(0.05)

    print("[ROUTINE] Exiting Target Drop routine.")

    

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
            print("RUNNING TARGET DROP ROUTINE")
            routine_target_drop()

        elif mode == 2:
            sleep(0.1)
            continue

        elif mode == 3:
            goto_coordinate(targetDropTestCoordinate.latitude, targetDropTestCoordinate.longitude, targetDropTestCoordinate.altitude)
            lawnmowerSearch(local_pts, ref_lat, ref_lon, SEARCH_ALTITUDE, spacingBetweenPaths=5.0)
            sleep(0.1)
            continue

        elif mode == 4:
            sleep(0.1) 
            continue

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
    
    global last_rc_print_time

    if KEYBOARD_TEST_MODE:
        return
    remote_control_5 = msg.chan5_raw  # SA
    remote_control_6 = msg.chan6_raw  # SC
    manual_drop = msg.chan7_raw       # SD
    flight_mode_switch = msg.chan8_raw  # SB

    current_time = time()

    if current_time - last_rc_print_time >= SERIAL_PRINT_INTERVAL:
        print(
            f"[RC] CH5 ={remote_control_5}, | "
            f"CH6 ={remote_control_6}, | "
            f"CH7 ={manual_drop}, | "
            f"CH8 ={flight_mode_switch}"
        )
        last_rc_print_time = current_time

    # Manual payload drop
    if manual_drop > 1800:
        if try_drop_payload():
            print("[MANUAL DROP] Payload drop triggered.")

    # SB low: emergency/recovery RTL
    if flight_mode_switch < 1300:
        request_auto_mode(0)
        set_mode("LOITER")
        return

    # SB middle: pilot controlled AltHold
    if flight_mode_switch > 1700:
        request_auto_mode(0)
        set_mode("RTL")  # or ALT_HOLD if you want to allow manual stick control, but GUIDED is safer for switching in and out of autonomous modes
        return
    
    requested_mode = 0

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

     # SB high: companion computer allowed
    request_auto_mode(requested_mode)

    if requested_mode == 0:
        set_mode("LOITER")
    else:
        set_mode("GUIDED")


def mavlink_loop():
    global master
    global last_position_print_time
    global last_heartbeat_print_time
    global dronePosition

    # Opens a serial USB connection between the Pixhawk and Raspberry Pi
    master = mavutil.mavlink_connection("tcp:192.168.1.4:5762")

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
                master = mavutil.mavlink_connection("tcp:192.168.1.4:5762")
                master.wait_heartbeat()

            print("[MAVLINK] Reconnected.")
            continue

        if msg_type == "GLOBAL_POSITION_INT":
            dronePosition["latitude"] = msg.lat / 1e7
            dronePosition["longitude"] = msg.lon / 1e7
            dronePosition["altitude"] = msg.relative_alt / 1000.0

            current_time = time()

            if current_time - last_position_print_time >= SERIAL_PRINT_INTERVAL:
                print(
                    f"Current Position: "
                    f"lat={dronePosition['latitude']}, "
                    f"lon={dronePosition['longitude']}, "
                    f"alt={dronePosition['altitude']}"
                )
                print("------------------------------------")
                last_position_print_time = current_time

        elif msg_type == "HEARTBEAT":
            actual_mode = mavutil.mode_string_v10(msg).upper()
            current_time = time() 

            if current_time - last_heartbeat_print_time >= SERIAL_PRINT_INTERVAL:
                print(f"Heartbeat: Current Pixhawk mode: {actual_mode}")
                last_heartbeat_print_time = current_time
            
        elif msg_type == "RC_CHANNELS":
            handle_rc_channels(msg)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=gstreamer_loop, daemon=True).start()
    threading.Thread(target=mavlink_loop, daemon=True).start()
    threading.Thread(target=autonomy_loop, daemon=True).start()

    if KEYBOARD_TEST_MODE:
        threading.Thread(target=keyboard_test_loop, daemon=True).start()

    while True:
        sleep(1)