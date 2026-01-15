# Obstacle Detection System 🚗

A local web app that lets you upload a driving video and returns an **annotated video** with **bounding boxes** for detected obstacles using **YOLOv8** deep learning model.

## 🆕 New Features (YOLOv8 Upgrade)

- **Deep Learning Detection**: Uses YOLOv8 for accurate obstacle detection
- **Object Classification**: Identifies specific objects (car, person, truck, bus, motorcycle, bicycle, traffic light, stop sign, animals...)
- **Confidence Score**: Shows detection confidence percentage
- **Color-coded Boxes**: Different colors for different obstacle types
- **Fallback Mode**: Automatically falls back to basic motion detection if YOLO unavailable

---

## Overview

The app provides a simple end-to-end flow:

- Upload a video from the browser.
- Backend (FastAPI) processes frames with OpenCV.
- Backend generates a new video with overlays (yellow boxes + `obstacle` label).
- Frontend (Next.js) previews the annotated video and allows download.

---

## Features

- **Video upload** from the web UI.
- **Annotated output video (MP4)** returned by the API.
- **Preview & download** annotated results in the browser.
- Local development scripts to run backend + frontend together.

---

## Tech Stack

- **Frontend**: Next.js (React)
- **Backend**: FastAPI (Python)
- **Computer Vision**: OpenCV
- **Dev tooling**: concurrently (run FE/BE together)

---

## Project Structure

```
.
├─ backend/
│  ├─ app/
│  │  ├─ main.py            # FastAPI app (upload endpoint)
│  │  └─ vision.py           # OpenCV pipeline + video annotation
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

---

## Setup (Local)

From the repository root:

```bash
npm install
npm run setup
```

What `npm run setup` does:

- Creates `backend/.venv`
- Installs Python dependencies from `backend/requirements.txt`
- Installs frontend dependencies in `frontend/`

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

1.  Open http://localhost:3000
2.  Choose a video file (`.mp4`, `.avi`, `.mov`, `.mkv`)
3.  Click **Upload & Analyze**
4.  Wait for processing, then preview / download the annotated video

---

## API

- **POST** `/api/upload`
  - Content-Type: `multipart/form-data`
  - Field: `file` (video)
  - Response: `video/mp4` (annotated output)

---

## Detection Modes

### YOLOv8 Mode (Default)

- **Detectable Objects**: car, person, truck, bus, motorcycle, bicycle, traffic light, stop sign, dog, cat, horse, and more
- **Confidence Threshold**: 50% (configurable)
- **Color Coding**:
  - 🟢 Green: person
  - 🔴 Red: car, stop sign
  - 🟣 Magenta: motorcycle
  - 🟡 Yellow: traffic light
  - 🔵 Cyan: bus, truck

### Basic Mode (Fallback)

If YOLO is unavailable, the system falls back to motion-based detection:

- Uses OpenCV background subtraction
- Labels all detected motion as "obstacle"
- Less accurate but works without GPU

---

## Troubleshooting

- **Push to GitHub failed because of large files**:

  - Do not commit `frontend/node_modules` or `backend/.venv`.
  - If they were committed before, remove them from tracking using `git rm -r --cached ...`.

- **Backend cannot write MP4 (`Cannot open video writer`)**:

  - This is usually a codec issue on Windows/OpenCV. Consider switching output to AVI (MJPG/XVID) in `vision.py`.

- **`Cannot open video`** when uploading:
  - Try a different video format/codec (e.g., MP4 H.264).

---

## Roadmap (Ideas)

- Better visualization (confidence, counts, timeline)
- Export annotated frames / snapshots
- Replace motion detection with a trained object detector (e.g., YOLO)

---

## Contributing

Pull requests are welcome. Please open an issue first to discuss major changes.

---

## License

Add a license if you plan to publish/redistribute this project.
