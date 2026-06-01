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
    modes = master.mode_mapping()
    if mode_name not in modes:
        raise Exception(f"Mode {mode_name} not available. Available modes: {list(modes.keys())}")

    mode_id = modes[mode_name]

    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )

    print(f"Mode set command sent: {mode_name}")

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

    start_time = time.time()

    while time.time() - start_time < seconds:
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

def has_position():
    return (
        dronePosition["latitude"] is not None and
        dronePosition["longitude"] is not None and
        dronePosition["altitude"] is not None
    )


def set_flight_mode(mode):
    global master

    if master is None:
        print("MAVLink connection not established.")
        return False

    mode = mode.upper()

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
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
            0, 0, 0, 0, 0
        )

    print(f"[MODE] Requested mode change to {mode} ({mode_id})")
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
        sleep(0.1)

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
                alt,  # altitude in m
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


def wait_until_reached(target_lat, target_lon, target_alt, radius=2.0, timeout=60):
    """Waits until the drone reaches a specified location, while periodically resending the target."""
    start = time()
    last_resend = 0

    while time() - start < timeout:
        if AUTO_MODE == 0:
            print("[WAYPOINT] Cancelled by manual mode.")
            return False

        # Resend waypoint every 2 seconds
        if time() - last_resend > 2.0:
            goto_location(target_lat, target_lon, target_alt)
            last_resend = time()

        lat = dronePosition["latitude"]
        lon = dronePosition["longitude"]

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
                request_pixhawk_mode("LOITER")
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
                    

                    AUTO_MODE = 0
                    request_pixhawk_mode("LOITER")
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
    request_pixhawk_mode("LOITER")

def routine_target_localization():
    """
    AUTO_MODE 3:
    Moves through a lawnmower search grid inside the geofence.
    While moving, it constantly checks for Bullseye.
    When Bullseye is detected, it records the drone's current GPS position.
    """

    global AUTO_MODE

    print("[ROUTINE] Target Localization lawnmower search started.")

    if not has_position():
        print("[LOCALIZE] No GPS position yet. Cannot start localization search.")
        AUTO_MODE = 0
        return

    request_pixhawk_mode("GUIDED")
    print(f"[LOCALIZE] Generated {len(LOCALIZATION_SEARCH_WAYPOINTS)} search waypoints.")


    for index, wp in enumerate(LOCALIZATION_SEARCH_WAYPOINTS, start=1):
        if AUTO_MODE != 3:
            print("[LOCALIZE] Localization search cancelled.")
            return

        print(f"[LOCALIZE] Going to search waypoint {index}/{len(LOCALIZATION_SEARCH_WAYPOINTS)}")
        print(f"[LOCALIZE] lat={wp.latitude}, lon={wp.longitude}, alt={wp.altitude}")

        goto_location(wp.latitude, wp.longitude, wp.altitude)

        waypoint_start = time()
        last_resend = 0

        while AUTO_MODE == 3:
            # Keep resending the current waypoint every 2 seconds
            if time() - last_resend > 2.0:
                goto_location(wp.latitude, wp.longitude, wp.altitude)
                last_resend = time()

            # Run vision while traveling
            detection = detect_target()

            if detection is not None:
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

            # Check if waypoint reached
            current_lat = dronePosition["latitude"]
            current_lon = dronePosition["longitude"]

            if current_lat is not None and current_lon is not None:
                distance = distance_meters(
                    current_lat,
                    current_lon,
                    wp.latitude,
                    wp.longitude
                )

                print(f"[LOCALIZE] Distance to search waypoint: {distance:.2f} meters")

                if distance <= 2.0:
                    print(f"[LOCALIZE] Reached search waypoint {index}.")
                    break

            # Timeout for this waypoint
            if time() - waypoint_start > 60:
                print(f"[LOCALIZE] Timeout on search waypoint {index}. Moving to next.")
                break

            sleep(0.1)

    print("[LOCALIZE] Finished lawnmower search. Switching to LOITER.")
    AUTO_MODE = 0
    request_pixhawk_mode("LOITER")
    
def routine_waypoint_navigation():
    global AUTO_MODE

    print("[ROUTINE] Waypoint Navigation started.")

    if not has_position():
        print("[WAYPOINT] No GPS position yet. Cannot start waypoint navigation.")
        AUTO_MODE = 0
        return

    request_pixhawk_mode("GUIDED")

    for index, wp in enumerate(NAV_WAYPOINTS, start=1):
        if AUTO_MODE != 4:
            print("[WAYPOINT] Cancelled.")
            return

        print(f"[WAYPOINT] Going to waypoint {index}/{len(NAV_WAYPOINTS)}")
        print(f"[WAYPOINT] lat={wp.latitude}, lon={wp.longitude}, alt={wp.altitude}")

        goto_location(wp.latitude, wp.longitude, wp.altitude)

        reached = wait_until_reached(
            wp.latitude,
            wp.longitude,
            wp.altitude,
            radius=2.0,
            timeout=60,
        )

        if not reached:
            print(f"[WAYPOINT] Failed to reach waypoint {index}. Moving on.")
        else:
            print(f"[WAYPOINT] Reached waypoint {index}.")

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
    global last_rc_print_time

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
        request_pixhawk_mode("LOITER")
        return

    # SB middle: pilot controlled AltHold
    if flight_mode_switch > 1700:
        request_auto_mode(0)
        request_pixhawk_mode("RTL")  # or ALT_HOLD if you want to allow manual stick control, but GUIDED is safer for switching in and out of autonomous modes
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
        request_pixhawk_mode("LOITER")
    else:
        request_pixhawk_mode("GUIDED")


def mavlink_loop():
    global master
    global last_position_print_time
    global last_heartbeat_print_time
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
    #threading.Thread(target=mavlink_loop, daemon=True).start()
    #threading.Thread(target=autonomy_loop, daemon=True).start()

    while True:
        sleep(1)