# Phân Tích Chi Tiết Hệ Thống Phát Hiện Chướng Ngại Vật

## (Obstacle Detection System - Đa Cấp Độ Kỹ Thuật)

---

## 📋 Tổng Quan Các Phương Pháp

| Cấp độ         | Phương pháp                    | Độ phức tạp       | Yêu cầu phần cứng           | Độ chính xác |
| -------------- | ------------------------------ | ----------------- | --------------------------- | ------------ |
| **Cơ bản**     | OpenCV (Contours, Canny)       | ⭐ Thấp           | Raspberry Pi, Camera thường | Trung bình   |
| **Nâng cao**   | Deep Learning (YOLO)           | ⭐⭐⭐ Trung bình | GPU hoặc CPU mạnh           | Cao          |
| **Chuyên sâu** | Monocular Depth (PyDNet/MiDaS) | ⭐⭐⭐⭐⭐ Cao    | GPU khuyến nghị             | Rất cao      |

---

## 1. 🔧 CẤP ĐỘ CƠ BẢN: OpenCV (Xử Lý Ảnh Truyền Thống)

### 1.1 Nguồn tham khảo

| Nguồn                                                                                                                       | Mô tả                                  |
| --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| [LearnOpenCV - Contour Detection](https://learnopencv.com/contour-detection-using-opencv-python-c/)                         | Hướng dẫn phát hiện đường bao chi tiết |
| [SihabSahariar/Rover-Navigation](https://github.com/SihabSahariar/Computer-Vision-Based-Rover-Navigation-Avoiding-Obstacle) | Robot tránh vật cản sử dụng OpenCV     |

### 1.2 Ứng dụng của Contours trong Computer Vision

```
┌─────────────────────────────────────────────────────────────────┐
│                    ỨNG DỤNG CONTOUR DETECTION                   │
├─────────────────────────────────────────────────────────────────┤
│  • Phát hiện chuyển động (Motion Detection)                    │
│  • Phát hiện vật thể bị bỏ quên (Unattended Object Detection)  │
│  • Phân tách nền/vật thể (Background/Foreground Segmentation)  │
│  • Nhận dạng hình dạng (Shape Recognition)                     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Contour là gì?

- **Contour** = Đường nối tất cả các điểm trên biên của một vật thể
- Các điểm có cùng **màu sắc** và **cường độ** pixel
- OpenCV cung cấp 2 hàm chính:
  - `findContours()` - Tìm contours
  - `drawContours()` - Vẽ contours

### 1.4 Các bước phát hiện Contour

```python
import cv2

# BƯỚC 1: Đọc ảnh và chuyển sang Grayscale
image = cv2.imread('input/image.jpg')
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# BƯỚC 2: Áp dụng Binary Thresholding
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

# BƯỚC 3: Tìm Contours
contours, hierarchy = cv2.findContours(
    thresh,
    cv2.RETR_TREE,           # Retrieval mode
    cv2.CHAIN_APPROX_SIMPLE  # Approximation method
)

# BƯỚC 4: Vẽ Contours lên ảnh gốc
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Hiển thị
cv2.imshow('Contours', image)
cv2.waitKey(0)
```

### 1.5 Các thuật toán Contour Approximation

| Thuật toán            | Mô tả                                | Số điểm lưu           |
| --------------------- | ------------------------------------ | --------------------- |
| `CHAIN_APPROX_NONE`   | Lưu TẤT CẢ điểm biên                 | Nhiều                 |
| `CHAIN_APPROX_SIMPLE` | Chỉ lưu điểm đầu cuối của đoạn thẳng | Ít (Tiết kiệm bộ nhớ) |

### 1.6 Các Retrieval Mode

| Mode            | Mô tả                                    |
| --------------- | ---------------------------------------- |
| `RETR_EXTERNAL` | Chỉ lấy contour ngoài cùng               |
| `RETR_LIST`     | Lấy tất cả contours, không phân cấp      |
| `RETR_TREE`     | Lấy tất cả với cấu trúc phân cấp cha-con |
| `RETR_CCOMP`    | 2 cấp: ngoài và lỗ bên trong             |

### 1.7 Code Rover tránh vật cản (Tham khảo)

```python
# Cài đặt
# pip install opencv-python numpy

import cv2
import numpy as np

def detect_obstacle(frame):
    """Phát hiện chướng ngại vật bằng màu sắc hoặc hình dạng"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Phát hiện cạnh
    edges = cv2.Canny(blur, 50, 150)

    # Tìm contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Lọc theo diện tích
            x, y, w, h = cv2.boundingRect(cnt)
            obstacles.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return frame, obstacles
```

---

## 2. 🚀 CẤP ĐỘ NÂNG CAO: Deep Learning (YOLO)

### 2.1 Nguồn tham khảo

| Nguồn                                                                                                       | Mô tả                                         |
| ----------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| [sailee2781/obstacle_detection_recognition-](https://github.com/sailee2781/obstacle_detection_recognition-) | YOLO v5 cho xe tự hành + ước tính khoảng cách |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)                                            | Thư viện YOLO chính thức, mới nhất            |

### 2.2 YOLO là gì?

**YOLO (You Only Look Once)** - Thuật toán phát hiện vật thể real-time mạnh mẽ nhất hiện nay.

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUY TRÌNH YOLO                               │
├─────────────────────────────────────────────────────────────────┤
│  Ảnh đầu vào → Neural Network → Bounding Boxes + Class Labels  │
│                                                                 │
│  ✓ Một lần chạy = Phát hiện TẤT CẢ vật thể                     │
│  ✓ Tốc độ: 30-60+ FPS (real-time)                              │
│  ✓ Độ chính xác cao                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Các phiên bản YOLO

| Version | Năm  | Đặc điểm                            |
| ------- | ---- | ----------------------------------- |
| YOLOv3  | 2018 | Ổn định, tài liệu phong phú         |
| YOLOv5  | 2020 | Dễ sử dụng, PyTorch                 |
| YOLOv8  | 2023 | Mới nhất, tốc độ + độ chính xác cao |
| YOLO26  | 2025 | Phiên bản mới nhất từ Ultralytics   |

### 2.4 Cài đặt Ultralytics YOLO

```bash
pip install ultralytics
```

### 2.5 Code YOLO cơ bản

```python
from ultralytics import YOLO

# 1. Load model pretrained
model = YOLO("yolov8n.pt")  # n=nano, s=small, m=medium, l=large, x=extra-large

# 2. Phát hiện vật thể trên ảnh
results = model("path/to/image.jpg")

# 3. Hiển thị kết quả
results[0].show()

# 4. Hoặc xử lý kết quả
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Tọa độ
        conf = box.conf[0]             # Độ tin cậy
        cls = box.cls[0]               # Class ID
        print(f"Class: {cls}, Confidence: {conf:.2f}")
```

### 2.6 Code YOLO với Video/Webcam

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện vật thể
    results = model(frame, stream=True)

    for result in results:
        # Vẽ boxes lên frame
        annotated_frame = result.plot()
        cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 2.7 CLI Commands (Command Line Interface)

```bash
# Dự đoán trên ảnh
yolo predict model=yolov8n.pt source="image.jpg"

# Dự đoán trên video
yolo predict model=yolov8n.pt source="video.mp4"

# Dự đoán với webcam
yolo predict model=yolov8n.pt source=0

# Train custom dataset
yolo train model=yolov8n.pt data=custom_data.yaml epochs=100 imgsz=640
```

### 2.8 Các loại vật thể COCO Dataset (80 classes)

YOLO pretrained có thể nhận diện:

- **Phương tiện**: car, truck, bus, motorcycle, bicycle, boat, airplane, train
- **Người**: person
- **Động vật**: dog, cat, bird, horse, cow, sheep, elephant...
- **Đồ vật**: traffic light, stop sign, fire hydrant, bench, chair, tv, laptop...

### 2.9 Ước tính khoảng cách (Distance Estimation)

```python
# Công thức ước tính khoảng cách đơn giản
# Distance = (Known_Width * Focal_Length) / Pixel_Width

def estimate_distance(known_width, focal_length, pixel_width):
    """
    known_width: Chiều rộng thực của vật thể (cm)
    focal_length: Tiêu cự camera (pixels) - cần calibration
    pixel_width: Chiều rộng vật thể trong ảnh (pixels)
    """
    return (known_width * focal_length) / pixel_width

# Ví dụ: Ước tính khoảng cách đến xe
CAR_WIDTH = 180  # cm (chiều rộng trung bình của xe)
FOCAL_LENGTH = 700  # Cần calibrate cho camera cụ thể

# Trong vòng lặp detection
for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    pixel_width = x2 - x1
    distance = estimate_distance(CAR_WIDTH, FOCAL_LENGTH, pixel_width)
    print(f"Distance: {distance:.2f} cm")
```

---

## 3. 🔬 CẤP ĐỘ CHUYÊN SÂU: Monocular Depth Estimation

### 3.1 Nguồn tham khảo

| Nguồn                                                                                                                 | Mô tả                          |
| --------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| [dronefreak/dji-tello-collision-avoidance-pydnet](https://github.com/dronefreak/dji-tello-collision-avoidance-pydnet) | PyDNet cho drone tránh va chạm |
| Paper: [Towards real-time unsupervised monocular depth estimation on CPU](https://arxiv.org/abs/1806.11430)           | IROS 2018                      |

### 3.2 Monocular Depth là gì?

```
┌─────────────────────────────────────────────────────────────────┐
│                 MONOCULAR DEPTH ESTIMATION                      │
├─────────────────────────────────────────────────────────────────┤
│  Ảnh 2D (1 camera) → Neural Network → Depth Map (Bản đồ chiều sâu)
│                                                                 │
│  ✓ Không cần camera chiều sâu (RGB-D) hoặc LiDAR                │
│  ✓ Ước tính khoảng cách từ camera đến mọi điểm trong ảnh       │
│  ✓ Ứng dụng: Drone, Robot, Xe tự hành                          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Tính năng dự án PyDNet

- 🚁 **Tello Drone Integration**: Real-time depth estimation
- 📷 **Webcam Support**: Test không cần drone
- 🧠 **PyDNet Depth Estimation**: Tối ưu cho CPU
- 🎯 **Collision Avoidance**: Navigation tự động
- ✅ **TensorFlow 2.x**: Modern framework

### 3.4 Cài đặt

```bash
# Clone repository
git clone https://github.com/dronefreak/dji-tello-collision-avoidance-pydnet.git
cd dji-tello-collision-avoidance-pydnet

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 3.5 Yêu cầu

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- (Optional) CUDA GPU cho tốc độ nhanh hơn

### 3.6 Cấu trúc dự án PyDNet

```
.
├── src/
│   ├── config.py              # Cấu hình
│   ├── depth_estimator.py     # Ước tính độ sâu PyDNet
│   ├── utils.py               # Utility functions
│   ├── camera_interface.py    # Abstract camera interface
│   ├── webcam_source.py       # Webcam implementation
│   ├── tello_source.py        # Tello drone
│   └── collision_avoidance.py # Logic tránh va chạm
├── tests/                     # Unit tests
├── webcam_demo.py             # Demo với webcam
├── tello_demo.py              # Demo với Tello
└── requirements.txt
```

### 3.7 Demo với Webcam

```bash
# Chạy demo không cần drone
python webcam_demo.py
```

---

## 4. 📊 SO SÁNH CÁC PHƯƠNG PHÁP

| Tiêu chí                 | OpenCV Basic | YOLO              | Depth Estimation |
| ------------------------ | ------------ | ----------------- | ---------------- |
| **Độ khó**               | ⭐ Dễ        | ⭐⭐⭐ Trung bình | ⭐⭐⭐⭐⭐ Khó   |
| **Tốc độ**               | Rất nhanh    | Nhanh (30+ FPS)   | Trung bình       |
| **Độ chính xác**         | Thấp         | Cao               | Rất cao          |
| **Nhận dạng class**      | ❌ Không     | ✅ 80+ classes    | ❌ Không         |
| **Ước tính khoảng cách** | ❌ Không     | ⚠️ Cần thêm logic | ✅ Có            |
| **Yêu cầu GPU**          | ❌ Không     | ⚠️ Khuyến nghị    | ⚠️ Khuyến nghị   |
| **Raspberry Pi**         | ✅ Tốt       | ⚠️ Cần tối ưu     | ❌ Khó chạy      |

---

## 5. 🎯 KHUYẾN NGHỊ CHO DỰ ÁN CỦA BẠN

### Nếu mới bắt đầu (Beginner):

```
📌 Chọn: OpenCV Contour Detection
   → Dễ hiểu, code ít, chạy mọi máy
```

### Nếu làm đồ án thực tế:

```
📌 Chọn: YOLOv8 với Ultralytics
   → Cân bằng tốt nhất giữa độ khó và hiệu quả
   → Chỉ vài dòng code là có kết quả
```

### Nếu cần ước tính khoảng cách chính xác:

```
📌 Chọn: YOLO + Depth Estimation kết hợp
   → YOLO để nhận dạng vật thể
   → Depth để tính khoảng cách
```

---

## 6. 🚀 KẾ HOẠCH TRIỂN KHAI ĐỀ XUẤT

### Phase 1: Cơ bản (Tuần 1)

- [ ] Setup môi trường Python + OpenCV
- [ ] Implement Contour Detection cơ bản
- [ ] Test với ảnh tĩnh

### Phase 2: YOLO Integration (Tuần 2)

- [ ] Cài đặt Ultralytics
- [ ] Implement phát hiện với YOLOv8
- [ ] Test với webcam real-time

### Phase 3: Nâng cao (Tuần 3)

- [ ] Thêm ước tính khoảng cách
- [ ] Tối ưu performance
- [ ] Thêm cảnh báo khi vật cản quá gần

### Phase 4: Hoàn thiện (Tuần 4)

- [ ] UI/UX improvements
- [ ] Documentation
- [ ] Testing & Bug fixes

---

## 7. 📚 TÀI LIỆU THAM KHẢO THÊM

### Papers:

1. [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo.pdf)
2. [YOLO9000: Better, Faster, Stronger](https://pjreddie.com/media/files/papers/YOLO9000.pdf)
3. [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
4. [Towards real-time unsupervised monocular depth estimation on CPU](https://arxiv.org/abs/1806.11430)

### Từ khóa tìm kiếm:

- `python obstacle detection yolo opencv`
- `monocular depth estimation obstacle avoidance`
- `autonomous vehicle obstacle detection github`
- `real-time object detection raspberry pi`

### Video hướng dẫn:

- **EdjeElectronics** - "Train YOLO Object Detection on Custom Data"
- **Train YOLOv8 on Custom Dataset** - Hướng dẫn huấn luyện model tùy chỉnh

---

## 8. 🔧 CÀI ĐẶT NHANH

```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Cài đặt các thư viện cần thiết
pip install opencv-python
pip install numpy
pip install matplotlib
pip install ultralytics  # Cho YOLO
pip install tensorflow   # Cho Depth Estimation
```

---

> **📌 Lưu ý**: File này tổng hợp từ nhiều nguồn để chuẩn bị cho việc triển khai hệ thống phát hiện chướng ngại vật. Khuyến nghị bắt đầu với **YOLOv8** để có kết quả nhanh nhất!
