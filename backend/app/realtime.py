from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2

from .vision import AnalyzeConfig, _detect_obstacles_basic, _detect_obstacles_yolo, _get_yolo_model, _resize_keep_aspect, _risk_level_for_bbox, YOLO_AVAILABLE


@dataclass
class RealtimeState:
    jpeg: Optional[bytes] = None
    frame_id: int = 0
    frame_width: int = 0
    frame_height: int = 0
    detections: list[dict[str, Any]] | None = None
    detection_mode: str = 'unknown'
    fps: float | None = None


class RealtimeService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = RealtimeState(detections=[])
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ref_count = 0

        self._cfg = AnalyzeConfig()
        self._src: str | int = 0

        self._last_infer_t = 0.0
        self._infer_fps = 0.0

    def configure(self, src: str | int, cfg: AnalyzeConfig) -> None:
        with self._lock:
            need_restart = src != self._src
            self._cfg = cfg
            self._src = src

        if need_restart:
            self._restart()

    def acquire(self) -> None:
        with self._lock:
            self._ref_count += 1
            should_start = self._thread is None or not self._thread.is_alive()
        if should_start:
            self._start()

    def release(self) -> None:
        with self._lock:
            self._ref_count = max(0, self._ref_count - 1)
            should_stop = self._ref_count == 0
        if should_stop:
            self._stop.set()

    def snapshot(self) -> RealtimeState:
        with self._lock:
            st = self._state
            return RealtimeState(
                jpeg=st.jpeg,
                frame_id=st.frame_id,
                frame_width=st.frame_width,
                frame_height=st.frame_height,
                detections=list(st.detections or []),
                detection_mode=st.detection_mode,
                fps=st.fps,
            )

    def _restart(self) -> None:
        self._stop.set()
        t = None
        with self._lock:
            t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        self._stop.clear()
        with self._lock:
            if self._ref_count > 0:
                self._start()

    def _start(self) -> None:
        self._stop.clear()
        t = threading.Thread(target=self._run, name='realtime_capture', daemon=True)
        with self._lock:
            self._thread = t
        t.start()

    def _run(self) -> None:
        cap: cv2.VideoCapture | None = None
        backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

        def open_cap() -> cv2.VideoCapture:
            src = self._src
            if isinstance(src, str) and src.isdigit():
                src = int(src)
            return cv2.VideoCapture(src)

        cap = open_cap()
        time.sleep(0.05)

        frame_index = -1
        last_detections: list[dict[str, Any]] = []
        last_w = 0
        last_h = 0

        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                try:
                    cap = open_cap()
                except Exception:
                    time.sleep(0.25)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            frame_index += 1
            cfg = self._cfg
            frame = _resize_keep_aspect(frame, cfg.resize_width)

            run_detection = True
            if cfg.sampled_every_n_frames > 1:
                run_detection = frame_index % cfg.sampled_every_n_frames == 0

            use_yolo = YOLO_AVAILABLE and _get_yolo_model() is not None
            if run_detection:
                if use_yolo:
                    raw = _detect_obstacles_yolo(frame, cfg)
                else:
                    raw = _detect_obstacles_basic(frame, cfg, backsub)

                fh = int(frame.shape[0])
                enriched = []
                for det in raw:
                    x, y, w, h = det['x'], det['y'], det['w'], det['h']
                    risk_level, reason = _risk_level_for_bbox(x, y, w, h, fh, cfg)
                    enriched.append(
                        {
                            'class_name': det.get('class_name'),
                            'class_id': det.get('class_id'),
                            'confidence': det.get('confidence'),
                            'bbox': {'x': x, 'y': y, 'w': w, 'h': h},
                            'risk_level': risk_level,
                            'reason': reason,
                        }
                    )
                last_detections = enriched

                now = time.time()
                if self._last_infer_t > 0:
                    dt = now - self._last_infer_t
                    if dt > 0:
                        inst = 1.0 / dt
                        self._infer_fps = 0.8 * self._infer_fps + 0.2 * inst
                self._last_infer_t = now

            h, w = frame.shape[:2]
            if w != last_w or h != last_h:
                last_w, last_h = w, h

            ok_jpg, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok_jpg:
                time.sleep(0.01)
                continue

            with self._lock:
                self._state.jpeg = buf.tobytes()
                self._state.frame_id += 1
                self._state.frame_width = int(w)
                self._state.frame_height = int(h)
                self._state.detections = list(last_detections)
                self._state.detection_mode = 'yolo' if use_yolo else 'basic'
                self._state.fps = float(self._infer_fps) if self._infer_fps > 0 else None

            time.sleep(0.001)

        try:
            cap.release()
        except Exception:
            pass
