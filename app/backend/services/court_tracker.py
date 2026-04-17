import cv2
import numpy as np

from .base_tracker import BaseTracker
from .court_estimator import CourtEstimator


class CourtTracker(BaseTracker):
    def __init__(self, local_model_path: str) -> None:
        super().__init__(local_model_path)
        self.estimator = CourtEstimator()
        print(f"CourtTracker loaded — task: {self.model.model.task}")

    def track_court(
        self,
        mp4_path: str,
        court_class_idx: int = 1,
        conf_thresh: float = 0.3,
    ) -> dict:
        avg_corners = None
        video_metadata = {"fps": 0, "width": 0, "height": 0, "total_frames": 0}

        try:
            cap = cv2.VideoCapture(mp4_path)
            video_metadata = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }
            self.estimator.court_class_id = court_class_idx

            # Allow ≤5% height difference between left/right corners of each side
            y_tolerance = video_metadata["height"] * 0.05
            sample_frames = [0, 29, 59, 89, 119]
            sampled_corners = []

            for frame_id in sample_frames:
                if frame_id >= video_metadata["total_frames"]:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if not ret:
                    continue

                estimation = self.estimator.predict(self.model(frame, verbose=False)[0])
                if estimation["status"] != "success" or estimation["confidence"] < conf_thresh:
                    continue

                corners = estimation["corners"]
                by_y = sorted(corners, key=lambda p: p[1])
                top, bottom = by_y[:2], by_y[2:]

                if (
                    abs(top[0][1] - top[1][1]) > y_tolerance
                    or abs(bottom[0][1] - bottom[1][1]) > y_tolerance
                ):
                    continue  # skip skewed / perspective-distorted detections

                tl, tr = sorted(top, key=lambda p: p[0])
                bl, br = sorted(bottom, key=lambda p: p[0])
                sampled_corners.append([tl, tr, br, bl])

            cap.release()

            if sampled_corners:
                avg_corners = np.array(sampled_corners).mean(axis=0).tolist()
                print(f"CourtTracker: estimated from {len(sampled_corners)} valid frames")
            else:
                print("CourtTracker: no valid court found in sampled frames")

        except Exception as e:
            print(f"CourtTracker error: {e}")

        return {"court_corners": avg_corners, "video_metadata": video_metadata}
