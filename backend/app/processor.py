from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from pathlib import Path
from threading import Thread
from typing import Any, Callable

from .job_store import JobStore
from .storage import ResultMeta, Storage
from .vision import AnalyzeConfig, annotate_video


ProgressCb = Callable[[int, int | None, str | None], None]


def start_job(
    *,
    job_store: JobStore,
    storage: Storage,
    job_id: str,
    input_path: Path,
    filename: str,
    cfg: AnalyzeConfig,
) -> None:
    t = Thread(
        target=_run_job,
        kwargs={
            "job_store": job_store,
            "storage": storage,
            "job_id": job_id,
            "input_path": input_path,
            "filename": filename,
            "cfg": cfg,
        },
        daemon=True,
    )
    t.start()


def _run_job(*, job_store: JobStore, storage: Storage, job_id: str, input_path: Path, filename: str, cfg: AnalyzeConfig) -> None:
    started = time.time()

    def progress_cb(processed: int, total: int | None, message: str | None) -> None:
        # Throttle updates to avoid excessive disk writes
        now = time.time()
        rec = job_store.get(job_id)
        if rec is None:
            return
        last = rec.updated_at
        if processed == 0 or (now - last) >= 0.25 or (total is not None and processed >= total):
            progress = 0.0
            if total and total > 0:
                progress = min(1.0, processed / total)
            job_store.update(
                job_id,
                status="running",
                progress=progress,
                processed_frames=processed,
                total_frames=total,
                message=message,
            )

    try:
        job_store.update(job_id, status="running", message="Starting")

        result_id = f"res_{uuid.uuid4().hex}"
        paths = storage.create_result_paths(result_id)

        events: list[dict[str, Any]] = []
        stats = annotate_video(
            input_path=str(input_path),
            output_path=str(paths.video_path),
            cfg=cfg,
            progress_cb=progress_cb,
            events_out=events,
            snapshots_dir=str(paths.snapshots_dir),
        )

        meta = ResultMeta(
            result_id=result_id,
            filename=filename,
            created_at=time.time(),
            processing_time_s=round(time.time() - started, 3),
            fps=stats.get("fps"),
            frame_count=stats.get("frame_count"),
            detection_mode=stats.get("detection_mode", "unknown"),
            config=asdict(cfg),
        )
        storage.write_json(paths.meta_path, meta.to_dict())
        storage.write_json(paths.events_path, {"result_id": result_id, "events": events})

        job_store.update(
            job_id,
            status="done",
            progress=1.0,
            message="Done",
            result_id=result_id,
            error=None,
        )
    except Exception as e:
        job_store.update(job_id, status="error", message="Error", error=str(e))
    finally:
        try:
            if input_path.exists():
                input_path.unlink()
        except Exception:
            pass
