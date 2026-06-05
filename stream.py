udp_url = "udp://0.0.0.0:5600"

cap = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Could not open UDP stream")
    exit()

print("Receiving UDP video... Press q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("No frame received")
        continue

    cv2.imshow("UDP Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()