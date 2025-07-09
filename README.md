# Helmet Detection App

Ứng dụng phát hiện người không đội mũ bảo hiểm theo thời gian thực, sử dụng mô hình YOLOv8 và giao diện tkinter.

## Dataset
https://drive.google.com/file/d/1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX/view

## Tính năng

- Phát hiện người không đội mũ bảo hiểm qua webcam.
- Giao diện đơn giản, dễ sử dụng với thư viện tkinter.
- Hiển thị FPS và trạng thái an toàn theo thời gian thực.
- Lưu ảnh chụp và video trong quá trình phát hiện.
- Tùy chọn camera và điều chỉnh mức confidence.

## Yêu cầu hệ thống

- Python 3.8 hoặc mới hơn
- Các thư viện Python cần thiết:
  - opencv-python
  - pillow
  - ultralytics

## Cài đặt

Tải mã nguồn từ GitHub:

```bash
git clone https://github.com/Sosarasan1/CV-project.git
cd CV-project
pip install -r requirements.txt

```

## Cấu trúc thư mục
```
CV-project/
│
├── models/
│   └── best.pt              # Mô hình YOLOv8 đã huấn luyện
│
├── videos/                  # Thư mục lưu video kết quả
│
│
├── src/
│   └── helmet_app.py        # Tệp chính để chạy ứng dụng
│
└── requirements.txt         # Danh sách các thư viện cần cài
```
## Cách chạy chương trình
```
cd src
python helmet_app.py
```

## Video demo:
https://youtu.be/R-a-AVyVv1I

