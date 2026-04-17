import time

import cv2
import numpy as np

from .base_tracker import BaseTracker
from utils.geometry import order_points


class ServeRecognizer(BaseTracker):
    def recognize_serves(
        self,
        mp4_path: str,
        court_corners: list = None,
        serve_class_idx: int = 0,
        conf_thresh: float = 0.7,
        cooldown_frames: int = 20,
    ) -> dict:
        serve_events = []
        video_metadata = {}

        middle_court_y = self._calc_middle_court_y(court_corners)

        try:
            cap = cv2.VideoCapture(mp4_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_metadata = {
                "fps": fps,
                "total_frames": total_frames,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }

            in_serve = False
            serve_start = 0
            serving_team = "Unknown"
            frames_since_positive = 0
            frame_id = 0
            start_time = time.time()
            last_log = start_time

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                is_positive = False
                serve_box = None
                for box in self.model(frame, verbose=False)[0].boxes:
                    if int(box.cls) == serve_class_idx and box.conf.item() >= conf_thresh:
                        is_positive = True
                        serve_box = box.xyxy[0].cpu().numpy()
                        break

                if is_positive:
                    if not in_serve:
                        in_serve = True
                        serve_start = frame_id
                        serving_team = self._classify_serving_team(serve_box, middle_court_y)
                    frames_since_positive = 0
                else:
                    if in_serve:
                        frames_since_positive += 1
                        if frames_since_positive > cooldown_frames:
                            in_serve = False
                            end_frame = frame_id - frames_since_positive
                            serve_events.append(self._make_event(serve_start, end_frame, fps, serving_team))
                            serving_team = "Unknown"

                frame_id += 1
                if fps > 0 and frame_id % round(fps) == 0:
                    elapsed = time.time() - last_log
                    remaining = (total_frames - frame_id) * (elapsed / round(fps))
                    print(f"  Serve [{frame_id}/{total_frames}] est. {remaining:.0f}s remaining")
                    last_log = time.time()

            if in_serve:
                serve_events.append(self._make_event(serve_start, frame_id - 1, fps, serving_team))

            cap.release()
            print(f"Serve Recognizer: {len(serve_events)} serves in {time.time() - start_time:.1f}s")

        except Exception as e:
            print(f"ServeRecognizer error: {e}")

        return {"serve_events": serve_events, "video_metadata": video_metadata}

    @staticmethod
    def _calc_middle_court_y(court_corners: list) -> float | None:
        if not court_corners or len(court_corners) != 4:
            return None
        try:
            ordered = order_points(np.array(court_corners, dtype="float32"))
            top_y = (ordered[0][1] + ordered[1][1]) / 2
            bottom_y = (ordered[2][1] + ordered[3][1]) / 2
            return (top_y + bottom_y) / 2
        except Exception as e:
            print(f"Could not compute middle court Y: {e}")
            return None

    @staticmethod
    def _classify_serving_team(serve_box: np.ndarray, middle_y: float | None) -> int | str:
        if middle_y is None or serve_box is None:
            return "Unknown"
        return 1 if serve_box[3] < middle_y else 0  # 1=far team, 0=near team

    @staticmethod
    def _make_event(start_frame: int, end_frame: int, fps: float, serving_team) -> dict:
        timestamp = start_frame / fps if fps > 0 else 0
        return {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "timestamp": round(timestamp, 2),
            "label": "serve",
            "serving_team": serving_team,
        }
