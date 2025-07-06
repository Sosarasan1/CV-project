
from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')  

    print("Starting model training...")
    results = model.train(data='data/data.yaml', 
                          epochs=50, 
                          imgsz=640, 
                          batch=8,
                          name='../helmet_yolov8s_run1') 

    print("Training finished.")
    print("Model and results are saved in the 'runs/detect/helmet_yolov8s_run1' directory.")

if __name__ == '__main__':
    main()