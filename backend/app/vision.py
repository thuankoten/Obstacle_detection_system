from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from typing import Callable, Optional

import cv2
import numpy as np

# Try to import YOLO, fallback to basic detection if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class AnalyzeConfig:
    sampled_every_n_frames: int = 1  # Process every frame for better accuracy
    min_contour_area: int = 800
    resize_width: int = 640
    confidence_threshold: float = 0.5  # Minimum confidence for YOLO detections
    roi_warning_y_ratio: float = 0.65
    roi_danger_y_ratio: float = 0.80
    # Classes considered as obstacles (COCO dataset class IDs)
    # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 
    # 9: traffic light, 11: stop sign, 16: dog, 17: cat
    obstacle_classes: List[int] = field(default_factory=lambda: [
        0,   # person
        1,   # bicycle
        2,   # car
        3,   # motorcycle
        5,   # bus
        7,   # truck
        9,   # traffic light
        11,  # stop sign
        13,  # bench
        16,  # dog
        17,  # cat
        18,  # horse
        19,  # sheep
        20,  # cow
    ])


# Global model instance (loaded once)
_yolo_model = None


def _get_yolo_model():
    """Load YOLO model (singleton pattern for efficiency)"""
    global _yolo_model
    if _yolo_model is None and YOLO_AVAILABLE:
        _yolo_model = YOLO("yolov8n.pt")  # Nano model - fast and lightweight
    return _yolo_model


def _resize_keep_aspect(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == 0:
        return frame
    if w == width:
        return frame
    new_h = int(h * (width / w))
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def _detect_obstacles_yolo(frame: np.ndarray, cfg: AnalyzeConfig) -> List[Dict[str, Any]]:
    """Detect obstacles using YOLOv8"""
    model = _get_yolo_model()
    if model is None:
        return []
    
    # Run inference
    results = model(frame, verbose=False, conf=cfg.confidence_threshold)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            
            # Only include specified obstacle classes
            if cls_id not in cfg.obstacle_classes:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            
            detections.append({
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "area": (x2 - x1) * (y2 - y1)
            })
    
    return detections


def _detect_obstacles_basic(frame: np.ndarray, cfg: AnalyzeConfig, backsub) -> List[Dict[str, Any]]:
    """Fallback detection using background subtraction (less accurate)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    fgmask = backsub.apply(gray)
    fgmask = cv2.medianBlur(fgmask, 5)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for c in contours:
        area = int(cv2.contourArea(c))
        if area < cfg.min_contour_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        detections.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "class_id": -1,
            "class_name": "obstacle",
            "confidence": 0.5,
            "area": area
        })
    
    return detections


# Color mapping for different obstacle types
OBSTACLE_COLORS = {
    "person": (0, 255, 0),      # Green
    "bicycle": (255, 165, 0),   # Orange
    "car": (0, 0, 255),         # Red
    "motorcycle": (255, 0, 255), # Magenta
    "bus": (255, 255, 0),       # Cyan
    "truck": (0, 165, 255),     # Orange-red
    "traffic light": (0, 255, 255),  # Yellow
    "stop sign": (0, 0, 255),   # Red
    "dog": (255, 200, 100),     # Light blue
    "cat": (200, 100, 255),     # Pink
    "default": (0, 255, 255),   # Yellow (default)
}


def _get_obstacle_color(class_name: str) -> tuple:
    """Get color for obstacle type"""
    return OBSTACLE_COLORS.get(class_name, OBSTACLE_COLORS["default"])


def _risk_level_for_bbox(x: int, y: int, w: int, h: int, frame_h: int, cfg: AnalyzeConfig) -> tuple[str, str | None]:
    bottom = y + h
    if frame_h <= 0:
        return "info", None

    bottom_ratio = bottom / frame_h
    if bottom_ratio >= cfg.roi_danger_y_ratio:
        return "danger", "near_bottom"
    if bottom_ratio >= cfg.roi_warning_y_ratio:
        return "warning", "enter_roi"
    return "info", None


def annotate_video(
    input_path: str,
    output_path: str,
    cfg: AnalyzeConfig,
    progress_cb: Optional[Callable[[int, int | None, str | None], None]] = None,
    events_out: Optional[list[dict[str, Any]]] = None,
    snapshots_dir: str | None = None,
) -> dict[str, Any]:
    """Process video, annotate detections, and optionally emit events + snapshots."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None

    # For fallback mode
    backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    
    use_yolo = YOLO_AVAILABLE and _get_yolo_model() is not None
    
    writer: cv2.VideoWriter | None = None
    frame_index = -1
    last_detections: List[Dict[str, Any]] = []
    wrote_snapshot_frames: set[int] = set()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            if progress_cb is not None:
                progress_cb(frame_index + 1, frame_count, "Processing")

            frame = _resize_keep_aspect(frame, cfg.resize_width)

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError("Cannot open video writer")

            # Run detection based on sampling rate
            run_detection = True
            if cfg.sampled_every_n_frames > 1:
                run_detection = frame_index % cfg.sampled_every_n_frames == 0

            if run_detection:
                if use_yolo:
                    last_detections = _detect_obstacles_yolo(frame, cfg)
                else:
                    last_detections = _detect_obstacles_basic(frame, cfg, backsub)

            # Draw detections + emit events
            fh = frame.shape[0]
            for det in last_detections:
                x, y, w, h = det["x"], det["y"], det["w"], det["h"]
                class_name = det["class_name"]
                conf = det["confidence"]

                risk_level, reason = _risk_level_for_bbox(x, y, w, h, fh, cfg)

                base_color = _get_obstacle_color(class_name)
                border_color = base_color
                if risk_level == "warning":
                    border_color = (0, 255, 255)
                elif risk_level == "danger":
                    border_color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2)

                label = f"{class_name} {conf:.0%}"
                if risk_level != "info":
                    label = f"{risk_level.upper()} | {label}"

                (label_w, label_h), _baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    frame,
                    (x, max(0, y - label_h - 10)),
                    (x + label_w + 4, y),
                    border_color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x + 2, max(label_h, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if events_out is not None and risk_level in {"warning", "danger"}:
                    ts_ms = int((frame_index / fps) * 1000) if fps and fps > 0 else 0
                    snapshot_name = None
                    if snapshots_dir is not None and frame_index not in wrote_snapshot_frames:
                        snapshot_name = f"{frame_index:06d}.jpg"
                        snapshot_path = os.path.join(snapshots_dir, snapshot_name)
                        cv2.imwrite(snapshot_path, frame)
                        wrote_snapshot_frames.add(frame_index)

                    events_out.append(
                        {
                            "timestamp_ms": ts_ms,
                            "frame_index": frame_index,
                            "class_name": class_name,
                            "confidence": conf,
                            "bbox": {"x": x, "y": y, "w": w, "h": h},
                            "risk_level": risk_level,
                            "reason": reason,
                            "snapshot": snapshot_name,
                        }
                    )
            
            # Draw detection mode indicator
            mode_text = "YOLO" if use_yolo else "Basic"
            cv2.putText(
                frame,
                f"Mode: {mode_text} | Obstacles: {len(last_detections)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if progress_cb is not None:
        progress_cb(frame_count or (frame_index + 1), frame_count, "Done")

    return {
        "fps": float(fps) if fps and fps > 0 else None,
        "frame_count": frame_count,
        "detection_mode": "yolo" if use_yolo else "basic",
    }


def analyze_video(video_path: str, cfg: AnalyzeConfig) -> dict:
    """Analyze video and return detection results as JSON"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None

    backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    use_yolo = YOLO_AVAILABLE and _get_yolo_model() is not None

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
        
        if use_yolo:
            detections = _detect_obstacles_yolo(frame, cfg)
        else:
            detections = _detect_obstacles_basic(frame, cfg, backsub)

        timestamp_ms = 0
        if fps and fps > 0:
            timestamp_ms = int((frame_index / fps) * 1000)

        frames_out.append({
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "boxes": detections,
        })

    cap.release()

    return {
        "fps": float(fps) if fps and fps > 0 else None,
        "frame_count": frame_count,
        "sampled_every_n_frames": cfg.sampled_every_n_frames,
        "detection_mode": "yolo" if use_yolo else "basic",
        "frames": frames_out,
    }
