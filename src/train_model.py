# train.py
from ultralytics import YOLO

def main():
    # --- 1. Load a Pre-trained Model ---
    # We use a pre-trained model (like 'yolov8s.pt' for 'small') as a starting point.
    # This is called Transfer Learning and is highly effective.
    model = YOLO('yolov8s.pt')  # 'n' for nano, 'm' for medium, 'l' for large

    # --- 2. Train the Model ---
    # The 'data' parameter points to our data.yaml file.
    # 'epochs' is the number of times the model will see the entire dataset. 50 is a good start.
    # 'imgsz' is the image size the model will be trained on. 640 is common.
    # 'batch' is the number of images processed at once. If you get memory errors, reduce this (e.g., to 4 or 2).
    print("Starting model training...")
    results = model.train(data='data/data.yaml', 
                          epochs=50, 
                          imgsz=640, 
                          batch=8,
                          name='../helmet_yolov8s_run1') # Optional: give your training run a custom name

    print("Training finished.")
    print("Model and results are saved in the 'runs/detect/helmet_yolov8s_run1' directory.")

if __name__ == '__main__':
    main()