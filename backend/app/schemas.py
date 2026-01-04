from __future__ import annotations

from pydantic import BaseModel


class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int
    area: int


class FrameDetections(BaseModel):
    frame_index: int
    timestamp_ms: int
    boxes: list[BoundingBox]


class VideoAnalysisResult(BaseModel):
    filename: str
    fps: float | None
    frame_count: int | None
    sampled_every_n_frames: int
    frames: list[FrameDetections]
