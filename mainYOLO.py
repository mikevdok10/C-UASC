from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
# flask is a python web framework to build applications easily 
from flask import Flask, Response, jsonify
import numpy as np
from pymavlink import mavutil
from gpiozero import AngularServo
from time import sleep, time
import threading 



trigger_drop = False
servo_busy = False
last_trigger_time = 0
TRIGGER_COOLDOWN = 5  

app = Flask(__name__)


telemetry = {
    "lat": 0.0,
    "lon": 0.0,
    "alt": 0.0,
    "mode": "UNKNOWN",
    "armed": False
}

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

master = None 

def mavlink_loop():
    #opens a serial UART connection between the pixhawk and the raspberry pi 
    global master   
    master = mavutil.mavlink_connection('/dev/serial0', baud=57600)

    # this bit just confirsms that the connection is active 
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print("Its alive....")
    

    # this will constatly wait for a MAVLink message to come through
    #the blocking just pauses until a new one arrives 
    while True:
            msg = master.recv_match(blocking=True)
            if not msg:
                continue


            # this filters out only the messages we care about 
            msg_type = msg.get_type()

            #tells us latitude, longitude, and altitude
            if msg_type == "GLOBAL_POSITION_INT":
                telemetry["lat"] = msg.lat / 1e7
                telemetry["lon"] = msg.lon / 1e7
                telemetry["alt"] = msg.relative_alt / 1000.0
            # checks if the drone is armed, and what flight state it is in (Loiter , auto, guided, etc) 
            elif msg_type == "HEARTBEAT":
                telemetry["mode"] = mavutil.mode_string_v10(msg)
                telemetry["armed"] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0


def drop_payload():
    global master
    if not master:
        print("MAVLink connection not established.")
        return  
    print ("dropping payload")
    trigger_servo(master, channel=8, pwm=1900)
    sleep(1)
    trigger_servo(master, channel=8, pwm=1100)  # reset position

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

            

threading.Thread(target=mavlink_loop, daemon=True).start()


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


def generate_frames():
    while True:
        frame = camera.capture_array()

        #camera settings are wierd, color correct before running detection 

        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = zoom_frame(frame, zoom_factor=2.0)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        # Run detector on the first frame that comes in 
        detected_objects = detector(frame, conf=0.01, verbose=False)

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
            global last_trigger_time, current_time
            global trigger_drop, last_trigger_time

            if class_name == "Bullseye" and confidence > 0.8:
                current_time = time() 
            
                if current_time - last_trigger_time > TRIGGER_COOLDOWN:
                    last_trigger_time = current_time
                    threading.Thread(target=drop_payload, daemon=True).start() # reset position

            # Draws box
            cv2.rectangle(
                labeled_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draws label 
            text_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            text_w, text_h = text_size
            label_y = max(y1 - 10, text_h + 10)

            cv2.rectangle(
                labeled_frame,
                (x1, label_y - text_h - 8),
                (x1 + text_w + 6, label_y + 4),
                (0, 255, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                labeled_frame,
                label,
                (x1 + 3, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        labeled_frame = cv2.resize(labeled_frame, (500, 500))

        ret, buffer = cv2.imencode('.jpg', labeled_frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

#returns an HTML page 
@app.route('/')
def index(): 
    return """
    <html>
    <body>

    <h2>Live Feed</h2>

    <!returns the steam of jpegs form the detection> 
    <img src="/video_feed" width="500">


    <!this returns the telemetry values from the pixhawk> 
    <h2>Telemetry</h2>
    <ul>
        <li>Latitude: <span id="lat">0</span></li>
        <li>Longitude: <span id="lon">0</span></li>
        <li>Altitude: <span id="alt">0</span></li>
        <li>Mode: <span id="mode">UNKNOWN</span></li>
        <li>Armed: <span id="armed">False</span></li>
    </ul>

    <!javascript code that updates the telemetry values on the browser window> 
    <script>
    // requests the data from flask
    let lastAlt = 0;
    async function updateTelemetry() {
        try {
        // this gets the telemetry values from erliler, in the format that is dispalyed at the top of the code
            const res = await fetch('/telemetry');
            const data = await res.json();

            // JS converts to usabel data 
            // this is smoothing out the altitude values, since they can be a bit jumpy, this just makes it so the altitude changes more gradually on the browser window
            if(Number.isFinite(data.alt)) {
            lastAlt = lastAlt * 0.9 + data.alt * 0.1;
            }

            // updates HTML text
            document.getElementById("lat").innerText =
                Number.isFinite(data.lat) ? data.lat.toFixed(6) : "0";

            document.getElementById("lon").innerText =
                Number.isFinite(data.lon) ? data.lon.toFixed(6) : "0";

            document.getElementById("alt").innerText =
                lastAlt.toFixed(2);

            document.getElementById("mode").innerText =
                data.mode ?? "UNKNOWN";

            document.getElementById("armed").innerText =
                data.armed ? "True" : "False";

        } catch (e) {
            console.log("Telemetry error:", e);
        }

    }

    // how many times to update the telemetry values per second, or 5 in this case 
updateTelemetry();
setInterval(updateTelemetry, 200);
    </script>

    </body>
    </html>
    """


# this returns a constant stream of JPEGs, where they replace each other over and over again
@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/telemetry')
def telemetry_data():
    return jsonify(telemetry)



if __name__ == '__main__':
    app.run(host='192.168.1.80', port=5000)