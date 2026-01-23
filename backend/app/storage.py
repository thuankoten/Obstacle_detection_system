from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ResultPaths:
    result_dir: Path
    meta_path: Path
    events_path: Path
    video_path: Path
    snapshots_dir: Path


class Storage:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.jobs_dir = root_dir / "jobs"
        self.results_dir = root_dir / "results"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def job_input_path(self, job_id: str, suffix: str) -> Path:
        return self.jobs_dir / f"{job_id}_input{suffix}"

    def create_result_paths(self, result_id: str) -> ResultPaths:
        result_dir = self.results_dir / result_id
        snapshots_dir = result_dir / "snapshots"
        result_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        return ResultPaths(
            result_dir=result_dir,
            meta_path=result_dir / "meta.json",
            events_path=result_dir / "events.json",
            video_path=result_dir / "annotated.mp4",
            snapshots_dir=snapshots_dir,
        )

    def write_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ResultMeta:
    result_id: str
    filename: str
    created_at: float
    processing_time_s: float
    fps: float | None
    frame_count: int | None
    detection_mode: str
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
