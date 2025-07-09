import tkinter as tk
from tkinter import Label, Button, Scale, filedialog, OptionMenu
from PIL import Image, ImageTk
import cv2
import time
from ultralytics import YOLO


model = YOLO("../models/best.pt")


# === List available cameras ===
def list_available_cameras(max_index=10):
    available = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.read()[0]:
            available.append(index)
        cap.release()
    return available

class HelmetApp:
    def __init__(self, window):
        self.window = window
        self.window.title("🪖 Helmet Detection Dashboard")
        self.window.geometry("1100x650")
        self.window.configure(bg="#f0f0f0")

       
        self.conf_threshold = tk.DoubleVar(value=0.5)
        camera_list = list_available_cameras(10)
        if not camera_list:
            camera_list = [0]
        self.camera_index = tk.IntVar(value=camera_list[0])

        self.cap = None
        self.running = False
        self.last_frame = None
        self.frame_count = 0
        self.start_time = time.time()

        
        self.out = None
        self.recording = True  

        # === Sidebar ===
        self.sidebar = tk.Frame(window, bg="#ffffff", width=220, relief="raised", bd=2)
        self.sidebar.pack(side="left", fill="y")

        Label(self.sidebar, text="Helmet App", font=("Helvetica", 20, "bold"), bg="#ffffff").pack(pady=20)

        Button(self.sidebar, text="▶️ Bắt đầu", command=self.start, width=20, bg="#4CAF50", fg="white").pack(pady=10)
        Button(self.sidebar, text="⏹ Dừng lại", command=self.stop, width=20, bg="#f44336", fg="white").pack(pady=10)
        Button(self.sidebar, text="💾 Chụp ảnh", command=self.save_snapshot, width=20, bg="#2196F3", fg="white").pack(pady=10)

        Label(self.sidebar, text="🎚 Confidence", bg="#ffffff").pack(pady=(30, 0))
        Scale(self.sidebar, variable=self.conf_threshold, from_=0.2, to=1.0, resolution=0.05,
              orient="horizontal", length=150).pack()

        Label(self.sidebar, text="📷 Camera Index", bg="#ffffff").pack(pady=(30, 0))
        OptionMenu(self.sidebar, self.camera_index, *camera_list).pack()

        self.status_label = Label(self.sidebar, text="", bg="#ffffff", fg="red", wraplength=180, font=("Arial", 11))
        self.status_label.pack(pady=30)

        # === Main Area ===
        self.main_area = tk.Frame(window, bg="#cccccc")
        self.main_area.pack(side="right", expand=True, fill="both")

        self.video_label = Label(self.main_area, bg="#000000")
        self.video_label.pack(expand=True)

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.camera_index.get(), cv2.CAP_DSHOW)
        self.running = True
        self.frame_count = 0
        self.start_time = time.time()

        
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.out = cv2.VideoWriter("recorded_output.avi", fourcc, 20.0, (640, 480))

        self.update_frame()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.out:
            self.out.release()
            self.out = None
        self.video_label.config(image='')
        self.status_label.config(text="")

    def save_snapshot(self):
        if self.last_frame is not None:
            filename = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                    filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
            if filename:
                cv2.imwrite(filename, self.last_frame)

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.window.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)

        try:
            results = model.predict(source=frame, imgsz=416, conf=self.conf_threshold.get(), stream=True)
            for r in results:
                annotated = r.plot()
                self.last_frame = annotated.copy()

                helmet_not_detected = any([box.cls == 0 for box in r.boxes]) if r.boxes else False
                if helmet_not_detected:
                    self.status_label.config(text="⚠️ Không phát hiện mũ bảo hiểm!")

                self.frame_count += 1
                fps = self.frame_count / (time.time() - self.start_time)
                cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Save video frame
                if self.out:
                    self.out.write(annotated)

        except Exception as e:
            print("🚨 Error:", e)
            annotated = self.last_frame if self.last_frame is not None else frame

        img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.imgtk = ImageTk.PhotoImage(img_pil)
        self.video_label.imgtk = self.imgtk
        self.video_label.config(image=self.imgtk)

        self.window.after(1, self.update_frame)

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = HelmetApp(root)
    root.mainloop()