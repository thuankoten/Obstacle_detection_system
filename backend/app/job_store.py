from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class JobRecord:
    job_id: str
    status: str  # queued|running|done|error
    progress: float
    processed_frames: int
    total_frames: int | None
    message: str | None
    result_id: str | None
    error: str | None
    created_at: float
    updated_at: float


class JobStore:
    def __init__(self, jobs_dir: Path):
        self._jobs_dir = jobs_dir
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, JobRecord] = {}

    def create_job(self) -> JobRecord:
        now = time.time()
        job_id = f"job_{uuid.uuid4().hex}"
        rec = JobRecord(
            job_id=job_id,
            status="queued",
            progress=0.0,
            processed_frames=0,
            total_frames=None,
            message="Queued",
            result_id=None,
            error=None,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._cache[job_id] = rec
            self._persist_locked(rec)
        return rec

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            rec = self._cache.get(job_id)
            if rec is not None:
                return rec

            path = self._jobs_dir / f"{job_id}.json"
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            rec = JobRecord(**data)
            self._cache[job_id] = rec
            return rec

    def update(self, job_id: str, **fields: Any) -> JobRecord:
        with self._lock:
            rec = self._cache.get(job_id)
            if rec is None:
                path = self._jobs_dir / f"{job_id}.json"
                if not path.exists():
                    raise KeyError(job_id)
                data = json.loads(path.read_text(encoding="utf-8"))
                rec = JobRecord(**data)
                self._cache[job_id] = rec

            for k, v in fields.items():
                if not hasattr(rec, k):
                    continue
                setattr(rec, k, v)
            rec.updated_at = time.time()
            self._persist_locked(rec)
            return rec

    def _persist_locked(self, rec: JobRecord) -> None:
        path = self._jobs_dir / f"{rec.job_id}.json"
        path.write_text(json.dumps(asdict(rec), ensure_ascii=False, indent=2), encoding="utf-8")
