from ultralytics import YOLO

# Load your trained model
model = YOLO('models/best.pt')

# Run detection on your test images
results = model.predict(source='helmet_dataset_final/test/images', save=True, conf=0.25)  # adjust conf if needed
