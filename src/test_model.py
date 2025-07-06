from ultralytics import YOLO

model = YOLO('../models/best.pt')

results = model.predict(source='helmet_dataset_final/test/images', save=True, conf=0.25)  
