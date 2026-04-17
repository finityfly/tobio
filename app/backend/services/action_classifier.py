import time
from collections import Counter, deque

import cv2

from .base_tracker import BaseTracker

_KNOWN_CLASSES = {0: "block", 1: "defense", 2: "serve", 3: "set", 4: "spike"}


class ActionClassifier(BaseTracker):
    def classify_action(
        self,
        mp4_path: str,
        conf_thresh: float = 0.5,
        sliding_window_size: int = 3,
        action_cooldowns: dict = None,
        default_cooldown: int = 15,
        trigger_count: int = 2,
    ) -> dict:
        if action_cooldowns is None:
            action_cooldowns = {}

        action_detections = []
        video_metadata = {"fps": 0, "width": 0, "height": 0, "total_frames": 0}

        try:
            cap = cv2.VideoCapture(mp4_path)
            video_metadata = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }

            all_cooldowns = list(action_cooldowns.values()) + [default_cooldown]
            max_cooldown = max(all_cooldowns)

            sliding_window = deque(maxlen=sliding_window_size)
            event_buffer = deque(maxlen=max_cooldown)

            active_type = None
            active_start = 0
            active_box = None
            frame_id = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.predict(frame, conf=conf_thresh, max_det=300, verbose=False)
                detected_names = []
                frame_boxes = {}

                if len(results[0]) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        name = _KNOWN_CLASSES.get(cls_id, f"unknown_{cls_id}")
                        detected_names.append(name)
                        frame_boxes[name] = box.xyxy[0].tolist()
                    sliding_window.extend(detected_names)

                    top = Counter(sliding_window).most_common(1)
                    if top:
                        cls_name, count = top[0]
                        if count >= trigger_count and cls_name is not None:
                            cooldown = action_cooldowns.get(cls_name, default_cooldown)
                            event_buffer.clear()
                            for _ in range(cooldown):
                                event_buffer.append((cls_name, frame_boxes.get(cls_name)))
                else:
                    sliding_window.append(None)

                current_action = current_box = None
                if event_buffer:
                    current_action, current_box = event_buffer.popleft()

                if current_action != active_type:
                    if active_type is not None:
                        action_detections.append({
                            "action": active_type,
                            "start_frame": active_start,
                            "end_frame": frame_id - 1,
                            "box": active_box,
                        })
                    if current_action is not None:
                        active_start = frame_id
                        active_box = current_box
                    active_type = current_action

                if active_type is not None and current_box is not None:
                    active_box = current_box

                frame_id += 1

            if active_type is not None:
                action_detections.append({
                    "action": active_type,
                    "start_frame": active_start,
                    "end_frame": frame_id - 1,
                    "box": active_box,
                })

            cap.release()

        except Exception as e:
            import traceback
            print(f"ActionClassifier error: {e}")
            traceback.print_exc()

        return {"action_detections": action_detections, "video_metadata": video_metadata}
