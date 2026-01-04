from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .vision import AnalyzeConfig, annotate_video

app = FastAPI(title="Obstacle Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> FileResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in_path = tmp_in.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp_in.write(chunk)

        fd, tmp_out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        cfg = AnalyzeConfig()
        annotate_video(tmp_in_path, tmp_out_path, cfg)

        background_tasks.add_task(_safe_remove, tmp_in_path)
        background_tasks.add_task(_safe_remove, tmp_out_path)

        return FileResponse(
            tmp_out_path,
            media_type="video/mp4",
            filename=f"annotated_{Path(file.filename).stem}.mp4",
            background=background_tasks,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
