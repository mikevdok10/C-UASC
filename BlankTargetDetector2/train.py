from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=75,
        batch=32,
        imgsz=768,
        workers=8,
        device=0
    )

if __name__ == "__main__":
    main()