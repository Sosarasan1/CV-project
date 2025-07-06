from ultralytics import YOLO

# Load model
model = YOLO("models/best.pt")

# Run on all images in 'self_test' folder, save results to 'outputs/helmet_results'
results = model.predict(
    source="self_test",       # Folder with your test images
    save=True,
    conf=0.5,
    project="outputs",        # Main output folder
    name="helmet_results",    # Subfolder name (overwrites if already exists)
    exist_ok=True             # Allow overwrite if it exists
)

from PIL import Image
import os

output_dir = "outputs/helmet_results"

for file in os.listdir(output_dir):
    if file.endswith(('.jpg', '.png')):
        image_path = os.path.join(output_dir, file)
        img = Image.open(image_path)
        img.show()  # This opens it with your default image viewer
