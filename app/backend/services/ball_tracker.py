import time

import cv2
import numpy as np

from .base_tracker import BaseTracker


class _BallKalmanFilter:
    """Constant-velocity 2D Kalman filter for volleyball tracking.

    State:       [cx, cy, vx, vy]
    Measurement: [cx, cy]
    """

    MAX_PREDICT = 30  # give up predicting after this many consecutive frames without a detection

    def __init__(self) -> None:
        kf = cv2.KalmanFilter(4, 2)

        # x_{t+1} = F * x_t  (constant velocity)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )

        # z_t = H * x_t  (we observe center only)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )

        # Q — process noise: higher → more agile (ball can change direction fast)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 3.0

        # R — measurement noise: lower → more trust in YOLO detections
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0

        kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0

        self._kf = kf
        self._initialized = False
        self._frames_since_detect = 0
        self._last_size: tuple[float, float] = (20.0, 20.0)

    def update(self, cx: float, cy: float, w: float, h: float) -> tuple[float, float]:
        """Feed a YOLO detection. Returns Kalman-corrected (cx, cy)."""
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        if not self._initialized:
            self._kf.statePre  = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
            self._kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
            self._initialized = True
        self._kf.correct(measurement)
        self._frames_since_detect = 0
        self._last_size = (w, h)
        state = self._kf.statePost
        return float(state[0]), float(state[1])

    def predict(self) -> tuple[float, float] | None:
        """Advance one frame. Returns predicted (cx, cy), or None if stale."""
        if not self._initialized:
            return None
        self._frames_since_detect += 1
        if self._frames_since_detect > self.MAX_PREDICT:
            return None
        state = self._kf.predict()
        return float(state[0]), float(state[1])

    @property
    def last_size(self) -> tuple[float, float]:
        return self._last_size


class BallTracker(BaseTracker):
    def track_ball(
        self, mp4_path: str, ball_class_idx: int = 0, conf_thresh: float = 0.15
    ) -> dict:
        """Track the volleyball through every frame using YOLO + Kalman filter.

        Lower conf_thresh improves recall; the Kalman filter smooths out any
        false positives by maintaining physical continuity (position + velocity).
        When YOLO misses the ball the filter predicts forward for up to
        _BallKalmanFilter.MAX_PREDICT consecutive frames before giving up.
        """
        ball_tracks = []
        video_metadata: dict = {"fps": 0, "width": 0, "height": 0, "total_frames": 0}
        kf = _BallKalmanFilter()

        try:
            cap = cv2.VideoCapture(mp4_path)
            video_metadata = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }
            fps = video_metadata["fps"]
            total = video_metadata["total_frames"]
            print(f"Ball Tracker: {total} frames @ {fps:.1f} fps")

            frame_id = 0
            start_time = time.time()
            last_log = start_time

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                detected = False
                for box in self.model(frame, verbose=False)[0].boxes:
                    if int(box.cls) == ball_class_idx and box.conf.item() >= conf_thresh:
                        b = box.xyxy.tolist()[0]
                        w, h = b[2] - b[0], b[3] - b[1]
                        raw_cx, raw_cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                        cx, cy = kf.update(raw_cx, raw_cy, w, h)
                        ball_tracks.append({
                            "frame": frame_id,
                            "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                            "confidence": box.conf.item(),
                        })
                        detected = True
                        break

                if not detected:
                    pos = kf.predict()
                    if pos is not None:
                        cx, cy = pos
                        w, h = kf.last_size
                        ball_tracks.append({
                            "frame": frame_id,
                            "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                            "confidence": 0.0,
                            "interpolated": True,
                        })

                frame_id += 1
                if fps > 0 and frame_id % round(fps) == 0:
                    elapsed = time.time() - last_log
                    remaining = (total - frame_id) * (elapsed / round(fps))
                    print(f"  Ball [{frame_id}/{total}] est. {remaining:.0f}s remaining")
                    last_log = time.time()

            cap.release()
            print(
                f"Ball Tracker: done in {time.time() - start_time:.1f}s "
                f"— {len(ball_tracks)} tracks ({sum(1 for t in ball_tracks if not t.get('interpolated'))} detected, "
                f"{sum(1 for t in ball_tracks if t.get('interpolated'))} predicted)"
            )

        except Exception as e:
            print(f"Ball Tracker error: {e}")

        return {"ball_tracks": ball_tracks, "video_metadata": video_metadata}
