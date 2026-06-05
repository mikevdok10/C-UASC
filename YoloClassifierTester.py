from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import numpy as np
from time import sleep

# Camera settings
STREAM_WIDTH = 640
STREAM_HEIGHT = 480
STREAM_FPS = 30

# Model paths
DETECTOR_PATH = "/home/bc/C-UASC/complete_detector_runs/detect/train2/weights/best_ncnn_model"
CLASSIFIER_PATH = "/home/bc/C-UASC/complete_classifier_runs/classify/train/weights/best_ncnn_model"

# Confidence settings
DETECT_CONFIDENCE = 0.25
CLASSIFY_CONFIDENCE = 0.6


def zoom_frame(frame, zoom_factor=1.0):
    h, w, _ = frame.shape

    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    cropped = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h))

    return zoomed


def get_classifier_label(class_results):
    probs = class_results[0].probs
    names = class_results[0].names

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


print("[TEST] Loading YOLO detector...")
detector = YOLO(DETECTOR_PATH)

print("[TEST] Loading YOLO classifier...")
classifier = YOLO(CLASSIFIER_PATH)

print("[TEST] Starting camera...")
camera = Picamera2()
camera.configure(
    camera.create_video_configuration(
        main={"size": (STREAM_WIDTH, STREAM_HEIGHT), "format": "RGB888"},
        controls={"FrameRate": STREAM_FPS}
    )
)
camera.start()

sleep(1)

print("[TEST] Running YOLO test. Press q to quit.")

while True:
    frame_rgb = camera.capture_array()

    if frame_rgb is None:
        continue

    # Convert RGB from Picamera2 to BGR for OpenCV
    frame_bgr = frame_rgb.copy()

    # Same zoom you use in your main drone code
    frame_bgr = zoom_frame(frame_bgr, zoom_factor=1.0)

    frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)

    # Run detector
    detected_objects = detector(frame_bgr, conf=DETECT_CONFIDENCE, verbose=False)

    if len(detected_objects) > 0:
        boxes = detected_objects[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detector_conf = float(box.conf[0])

            h, w = frame_bgr.shape[:2]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame_bgr[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Run classifier on detected crop
            class_results = classifier(crop, imgsz=224, verbose=False)
            class_name, class_conf = get_classifier_label(class_results)

            if class_conf < CLASSIFY_CONFIDENCE:
                continue

            label = f"{class_name} {class_conf:.2f} | det {detector_conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label background
            text_y = max(y1 - 10, 20)
            cv2.putText(
                frame_bgr,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            print(f"[VISION] {label} bbox=({x1}, {y1}, {x2}, {y2})")

    cv2.imshow("YOLO Detector + Classifier Test", frame_bgr)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.stop()
cv2.destroyAllWindows()
print("[TEST] Closed.")