import cv2
import time
import os
from ultralytics import YOLO

model = YOLO("../models/best.pt")

# Setup webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

# Setup video writer
save_dir = "runs/helmet_live_output"
os.makedirs(save_dir, exist_ok=True)
video_path = os.path.join(save_dir, "output.mp4")

# Get frame width and height
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_out = 20  # Change this if needed

# Define video codec and create writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(video_path, fourcc, fps_out, (width, height))

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    frame = cv2.flip(frame, 1)

    # Run detection
    results = model.predict(source=frame, conf=0.75, imgsz=416, stream=True)

    for r in results:
        annotated_frame = r.plot()

        #  Confusion Handling 
        helmet_detected = any([box.cls == 0 for box in r.boxes]) 
        if not helmet_detected: 
            cv2.putText(
                annotated_frame,
                "No helmet detected!",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        #  FPS Counter 
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        #  Save to video 
        out.write(annotated_frame)

        #  Display 
        cv2.imshow("Helmet Detection - Real-Time", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()  # Save and close the video file
cv2.destroyAllWindows()
