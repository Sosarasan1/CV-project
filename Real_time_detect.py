from ultralytics import YOLO

# Load your trained model
model = YOLO("models/best.pt")

# Use webcam as source (0 = default webcam)
model.predict(source=0, show=True, conf=0.5,imgsz=416,save=True)
