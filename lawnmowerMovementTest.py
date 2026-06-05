from pymavlink import mavutil
import time
import math

# For Mission Planner SITL, this is commonly correct:
master = mavutil.mavlink_connection("tcp:192.168.137.1:5762")

print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Connected to vehicle")


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

print("Local points:")
for p in local_pts:
    print(p)

# Put drone in GUIDED before sending movement commands.
set_mode("GUIDED")
time.sleep(1)

# Go to the first corner first.
goto_coordinate(
    boundingBoxCorners[0].latitude,
    boundingBoxCorners[0].longitude,
    boundingBoxCorners[0].altitude,
    seconds=5
)

# Start lawnmower search.
lawnmowerSearch(
    local_pts,
    ref_lat,
    ref_lon,
    altitude=2,
    spacingBetweenPaths=5
)
