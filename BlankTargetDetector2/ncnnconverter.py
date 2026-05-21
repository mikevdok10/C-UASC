from ultralytics import YOLO

def main():
    # Change this path to your trained model
    model = YOLO(r"C:\Users\peter\OneDrive\Desktop\BlankTargetDetector2\runs\detect\train2\weights\best.pt")

    model.export(
        format="ncnn",
        imgsz=640,
        half=False
    )

if __name__ == "__main__":
    main()