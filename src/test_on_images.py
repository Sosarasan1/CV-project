from ultralytics import YOLO

# Load model
model = YOLO("../models/best.pt")

results = model.predict(
    source="self_test",      
    save=True,
    conf=0.5,
    project="outputs",       
    name="helmet_results",  
    exist_ok=True             
)

from PIL import Image
import os

output_dir = "outputs/helmet_results"

for file in os.listdir(output_dir):
    if file.endswith(('.jpg', '.png')):
        image_path = os.path.join(output_dir, file)
        img = Image.open(image_path)
        img.show()  
