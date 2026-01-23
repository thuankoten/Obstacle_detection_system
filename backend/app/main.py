from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .job_store import JobStore
from .processor import start_job
from .realtime import RealtimeService
from .storage import Storage
from .vision import AnalyzeConfig

app = FastAPI(title="Obstacle Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)


_BACKEND_DIR = Path(__file__).resolve().parents[1]
_STORAGE = Storage(_BACKEND_DIR / "storage")
_JOB_STORE = JobStore(_STORAGE.jobs_dir)
_REALTIME = RealtimeService()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    sampled_every_n_frames: int = Form(1),
    confidence_threshold: float = Form(0.5),
    roi_warning_y_ratio: float = Form(0.65),
    roi_danger_y_ratio: float = Form(0.80),
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    job = _JOB_STORE.create_job()
    input_path = _STORAGE.job_input_path(job.job_id, suffix)

    try:
        with input_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        cfg = AnalyzeConfig(
            sampled_every_n_frames=max(1, int(sampled_every_n_frames)),
            confidence_threshold=float(confidence_threshold),
            roi_warning_y_ratio=float(roi_warning_y_ratio),
            roi_danger_y_ratio=float(roi_danger_y_ratio),
        )

        start_job(
            job_store=_JOB_STORE,
            storage=_STORAGE,
            job_id=job.job_id,
            input_path=input_path,
            filename=file.filename,
            cfg=cfg,
        )

        return JSONResponse({"job_id": job.job_id, "status": "queued"})
    except Exception as e:
        _JOB_STORE.update(job.job_id, status="error", error=str(e), message="Error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    rec = _JOB_STORE.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(
        {
            "job_id": rec.job_id,
            "status": rec.status,
            "progress": rec.progress,
            "processed_frames": rec.processed_frames,
            "total_frames": rec.total_frames,
            "message": rec.message,
            "result_id": rec.result_id,
            "error": rec.error,
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
        }
    )


@app.get("/api/results/{result_id}/meta")
def get_result_meta(result_id: str) -> JSONResponse:
    meta_path = _STORAGE.results_dir / result_id / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return JSONResponse(_STORAGE.read_json(meta_path))


@app.get("/api/results/{result_id}/events")
def get_result_events(result_id: str) -> JSONResponse:
    events_path = _STORAGE.results_dir / result_id / "events.json"
    if not events_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return JSONResponse(_STORAGE.read_json(events_path))


@app.get("/api/results/{result_id}/video")
def get_result_video(result_id: str) -> FileResponse:
    video_path = _STORAGE.results_dir / result_id / "annotated.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(
        str(video_path),
        media_type="video/mp4",
        filename=f"{result_id}.mp4",
    )


@app.get("/api/results/{result_id}/snapshots/{name}")
def get_result_snapshot(result_id: str, name: str) -> FileResponse:
    safe_name = Path(name).name
    snap_path = _STORAGE.results_dir / result_id / "snapshots" / safe_name
    if not snap_path.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(str(snap_path), media_type="image/jpeg")


@app.get("/api/realtime/stream")
def realtime_stream(
    src: str = "0",
    sampled_every_n_frames: int = 1,
    confidence_threshold: float = 0.5,
    roi_warning_y_ratio: float = 0.65,
    roi_danger_y_ratio: float = 0.80,
) -> StreamingResponse:
    cfg = AnalyzeConfig(
        sampled_every_n_frames=max(1, int(sampled_every_n_frames)),
        confidence_threshold=float(confidence_threshold),
        roi_warning_y_ratio=float(roi_warning_y_ratio),
        roi_danger_y_ratio=float(roi_danger_y_ratio),
    )
    _REALTIME.configure(src=src, cfg=cfg)
    _REALTIME.acquire()

    boundary = "frame"

    def gen():
        try:
            while True:
                st = _REALTIME.snapshot()
                if st.jpeg is None:
                    time.sleep(0.02)
                    continue

                yield (
                    f"--{boundary}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(st.jpeg)}\r\n\r\n"
                ).encode("utf-8") + st.jpeg + b"\r\n"
        finally:
            _REALTIME.release()

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


@app.websocket("/ws/realtime")
async def ws_realtime(
    websocket: WebSocket,
    src: str = "0",
    sampled_every_n_frames: int = 1,
    confidence_threshold: float = 0.5,
    roi_warning_y_ratio: float = 0.65,
    roi_danger_y_ratio: float = 0.80,
):
    await websocket.accept()

    cfg = AnalyzeConfig(
        sampled_every_n_frames=max(1, int(sampled_every_n_frames)),
        confidence_threshold=float(confidence_threshold),
        roi_warning_y_ratio=float(roi_warning_y_ratio),
        roi_danger_y_ratio=float(roi_danger_y_ratio),
    )

    _REALTIME.configure(src=src, cfg=cfg)
    _REALTIME.acquire()

    last_sent_frame_id = -1
    try:
        while True:
            st = _REALTIME.snapshot()
            if st.frame_id == last_sent_frame_id:
                await asyncio.sleep(0.03)
                continue
            last_sent_frame_id = st.frame_id

            await websocket.send_json(
                {
                    "frame_id": st.frame_id,
                    "frame_width": st.frame_width,
                    "frame_height": st.frame_height,
                    "detections": st.detections or [],
                    "detection_mode": st.detection_mode,
                    "fps": st.fps,
                }
            )
    except WebSocketDisconnect:
        return
    finally:
        _REALTIME.release()
