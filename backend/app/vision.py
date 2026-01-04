from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AnalyzeConfig:
    sampled_every_n_frames: int = 5
    min_contour_area: int = 800
    resize_width: int = 640


def _resize_keep_aspect(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == 0:
        return frame
    if w == width:
        return frame
    new_h = int(h * (width / w))
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def annotate_video(input_path: str, output_path: str, cfg: AnalyzeConfig) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    writer: cv2.VideoWriter | None = None
    frame_index = -1
    last_boxes: list[tuple[int, int, int, int]] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            frame = _resize_keep_aspect(frame, cfg.resize_width)

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError("Cannot open video writer")

            run_detection = True
            if cfg.sampled_every_n_frames > 1:
                run_detection = frame_index % cfg.sampled_every_n_frames == 0

            if run_detection:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                fgmask = backsub.apply(gray)
                fgmask = cv2.medianBlur(fgmask, 5)
                _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

                contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                boxes: list[tuple[int, int, int, int]] = []
                for c in contours:
                    area = int(cv2.contourArea(c))
                    if area < cfg.min_contour_area:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    boxes.append((int(x), int(y), int(w), int(h)))
                last_boxes = boxes

            for (x, y, w, h) in last_boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "obstacle",
                    (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def analyze_video(video_path: str, cfg: AnalyzeConfig) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None

    backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    frames_out: list[dict] = []
    frame_index = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1

        if cfg.sampled_every_n_frames > 1 and (frame_index % cfg.sampled_every_n_frames != 0):
            continue

        frame = _resize_keep_aspect(frame, cfg.resize_width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        fgmask = backsub.apply(gray)
        fgmask = cv2.medianBlur(fgmask, 5)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[dict] = []
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < cfg.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": area})

        timestamp_ms = 0
        if fps and fps > 0:
            timestamp_ms = int((frame_index / fps) * 1000)

        frames_out.append({
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "boxes": boxes,
        })

    cap.release()

    return {
        "fps": float(fps) if fps and fps > 0 else None,
        "frame_count": frame_count,
        "sampled_every_n_frames": cfg.sampled_every_n_frames,
        "frames": frames_out,
    }
