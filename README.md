# Obstacle Detection System 

A local web app for obstacle detection:

- **Detection Video**: upload a driving video, run analysis, and view results (annotated video + events)
- **Detection Real Time**: run detection from a local camera/webcam stream (MJPEG + WebSocket)

## Features

- **YOLOv8 detection** (with automatic fallback to basic motion detection if YOLO is unavailable)
- **Annotated output video** with bounding boxes (no class/conf text overlay)
- **Events** (warning/danger) with snapshots
- **Lane ROI filtering (trapezoid)**: only keep detections/events inside the current lane area
- **Realtime mode**: backend reads camera, frontend displays stream + bbox overlay

---

## Overview

This project is a monorepo:

- **Frontend**: Next.js (http://localhost:3000)
- **Backend**: FastAPI (http://127.0.0.1:8000)

---

## Tech Stack

- **Frontend**: Next.js (React)
- **Backend**: FastAPI (Python)
- **Computer Vision**: OpenCV
- **Deep Learning**: Ultralytics YOLOv8 (Torch)
- **Dev tooling**: concurrently (run FE/BE together)

---

## Project Structure

```
.
├─ backend/
│  ├─ app/
│  │  ├─ main.py            # FastAPI app (jobs + realtime endpoints)
│  │  ├─ realtime.py        # Realtime service (camera capture + detection)
│  │  └─ vision.py          # Video analysis + annotation
│  └─ requirements.txt
├─ frontend/
│  ├─ pages/                 # Next.js pages
│  └─ styles/
├─ package.json              # Root scripts (setup/dev)
└─ .gitignore
```

---

## Requirements

- **Python**: 3.10+ (recommended)
- **Node.js**: 18+ (recommended)
- **OS**: Windows (tested), should work on Linux/macOS with minor command changes

Optional:

- **GPU**: NVIDIA GPU + CUDA can improve YOLO performance (not required)

---

## Installation (Windows)

From the repository root:

```bash
npm install
npm run setup
```

What `npm run setup` does:

- Creates `backend/.venv`
- Installs Python dependencies from `backend/requirements.txt`
- Installs frontend dependencies in `frontend/`

If you want to setup manually:

```bash
# Backend
python -m venv backend/.venv
backend\.venv\Scripts\activate
pip install -r backend/requirements.txt

# Frontend
cd frontend
npm install
```

---

## Run (Local)

Start backend + frontend together:

```bash
npm run dev
```

- Frontend: http://localhost:3000
- Backend healthcheck: http://127.0.0.1:8000/health

---

## Usage

### 1) Detection Video

1. Open http://localhost:3000
2. Go to **Detection Video**
3. Upload a video file (`.mp4`, `.avi`, `.mov`, `.mkv`)
4. Configure parameters (sample rate / confidence / ROI thresholds / Lane ROI)
5. Start analysis and wait for completion
6. View the result video + events + snapshots

### 2) Detection Real Time

1. Go to **Detection Real Time**
2. Select `Camera index` (usually 0)
3. Adjust parameters + Lane ROI
4. Watch realtime stream and bbox overlay

---

## API

- **POST** `/api/jobs`
  - Content-Type: `multipart/form-data`
  - Fields:
    - `file`: video
    - `sampled_every_n_frames`, `confidence_threshold`, `roi_warning_y_ratio`, `roi_danger_y_ratio`
    - `lane_roi_enabled`, `lane_roi_center_x_ratio`, `lane_roi_top_y_ratio`, `lane_roi_bottom_y_ratio`, `lane_roi_top_width_ratio`, `lane_roi_bottom_width_ratio`

- **GET** `/api/jobs/{job_id}`
  - Poll job status/progress

- **GET** `/api/realtime/stream`
  - MJPEG stream
  - Query params match the same config fields (plus `src` for camera index)

- **WS** `/ws/realtime`
  - WebSocket stream of realtime detections/events

---

## Detection Modes

### YOLOv8 Mode (Default)

- **Detectable Objects**: car, person, truck, bus, motorcycle, bicycle, traffic light, stop sign, dog, cat, horse, and more
- **Confidence Threshold**: 50% (configurable)

### Basic Mode (Fallback)

If YOLO is unavailable, the system falls back to motion-based detection:

- Uses OpenCV background subtraction
- Labels all detected motion as "obstacle"
- Less accurate but works without GPU

---

## Troubleshooting

- **`npm run dev` fails**:
  - Ensure `npm install` succeeded at repo root
  - Ensure `npm run setup` completed (backend venv + requirements installed)
  - Check ports `3000` and `8000` are free

- **Push to GitHub failed because of large files**:

  - Do not commit `frontend/node_modules` or `backend/.venv`.
  - If they were committed before, remove them from tracking using `git rm -r --cached ...`.

- **Backend cannot write MP4 (`Cannot open video writer`)**:

  - This is usually a codec issue on Windows/OpenCV. Consider switching output to AVI (MJPG/XVID) in `vision.py`.

- **`Cannot open video`** when uploading:
  - Try a different video format/codec (e.g., MP4 H.264).

---

## Contributing

Pull requests are welcome. Please open an issue first to discuss major changes.

---

## License

Add a license if you plan to publish/redistribute this project.
