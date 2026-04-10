from ultralytics import YOLO

model = YOLO("yolov8n.pt") # loads the pre-trained YOLOv8n model    

model.export(format="ncnn") 