import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from scipy.spatial.distance import cdist
from torchvision.models import ResNet18_Weights, resnet18

from .base_tracker import BaseTracker

_REID_PREPROCESS = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PlayerTracker(BaseTracker):
    def __init__(self, local_model_path: str) -> None:
        super().__init__(local_model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_class_id = 0

        self.reid_model = self._load_reid_model()
        self.reid_preprocess = _REID_PREPROCESS

        # Per-video state (reset between track_players calls if needed)
        self._reset_state()

    def _load_reid_model(self) -> nn.Module:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
        model.to(self.device).eval()
        return model

    def _reset_state(self) -> None:
        self.player_gallery: dict = {}
        self.id_mapping: dict = {}
        self.next_unique_id: int = 1
        self.prev_frame_gray = None
        self.initial_assignment_done: bool = False
        self.best_init_frame: dict = {"frame": None, "detections": []}

    INITIALIZATION_WINDOW = 150

    def track_players(
        self,
        mp4_path: str,
        conf_thresh: float = 0.3,
        reid_sim_threshold: float = 0.25,
        max_unique_players: int = 12,
        stale_after_frames: int = 150,
    ) -> dict:
        self._reset_state()

        cap = cv2.VideoCapture(mp4_path)
        video_metadata = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        total = video_metadata["total_frames"]
        print(f"Player Tracker: cap={max_unique_players} | sim_thresh={reid_sim_threshold}")

        player_tracks = {}
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if self._detect_scene_change(frame):
                print(f"  Scene cut at frame {frame_id} — resetting spatial mapping")
                self.id_mapping = {}

            results = self.model.track(
                frame,
                persist=True,
                conf=conf_thresh,
                tracker="botsort.yaml",
                classes=[self.target_class_id],
                verbose=False,
            )

            if not self.initial_assignment_done:
                if frame_id <= self.INITIALIZATION_WINDOW:
                    self._update_best_init_frame(results, frame, video_metadata["height"], frame_id)
                else:
                    self._perform_initial_assignment(max_unique_players, frame_id)

            frame_detections = []
            if self.initial_assignment_done and results[0].boxes.id is not None:
                assigned = set()
                for box_obj, raw_yolo_id in zip(
                    results[0].boxes, results[0].boxes.id.int().cpu().tolist()
                ):
                    box = box_obj.xyxy.cpu().tolist()[0]
                    if not self._is_near_side_player(box, video_metadata["height"]):
                        continue
                    embedding = self._get_embedding(frame, box)
                    if embedding is None:
                        continue
                    pid = self._resolve_identity(
                        raw_yolo_id, embedding, assigned, reid_sim_threshold,
                        max_unique_players, box, video_metadata["width"],
                    )
                    if pid is not None:
                        assigned.add(pid)
                        frame_detections.append({
                            "player_id": pid,
                            "box": box,
                            "confidence": round(box_obj.conf.cpu().tolist()[0], 2),
                        })
                        gallery = self.player_gallery[pid]
                        gallery["embeddings"].append(embedding)
                        gallery["last_seen"] = frame_id
                        if len(gallery["embeddings"]) > 50:
                            gallery["embeddings"].pop(0)

            player_tracks[frame_id] = frame_detections

            if frame_id % 100 == 0 and self.initial_assignment_done:
                self._prune_stale_ids(frame_id, stale_after_frames)

            frame_id += 1
            if frame_id % 30 == 0:
                print(f"  Player [{frame_id}/{total}] | active IDs: {len(self.player_gallery)}")

        cap.release()
        return {"player_tracks": player_tracks, "video_metadata": video_metadata}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _detect_scene_change(self, frame, change_threshold: float = 0.40) -> bool:
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (128, 72))
        is_cut = False
        if self.prev_frame_gray is not None:
            diff = cv2.absdiff(gray, self.prev_frame_gray)
            if np.count_nonzero(diff > 30) > (gray.size * change_threshold):
                is_cut = True
        self.prev_frame_gray = gray
        return is_cut

    @staticmethod
    def _is_near_side_player(box: list, frame_height: int) -> bool:
        x1, y1, x2, y2 = box
        center_y = (y1 + y2) / 2
        height = y2 - y1
        return center_y >= frame_height * 0.35 and height >= frame_height * 0.08

    def _get_embedding(self, frame, box: list):
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        pad = 5
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        tensor = self.reid_preprocess(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.reid_model(tensor).cpu().numpy()

    @staticmethod
    def _centrality_penalty(box: list, frame_width: int, max_penalty: float = 0.15) -> float:
        deviation = abs((box[0] + box[2]) / 2 - frame_width / 2) / (frame_width / 2)
        return max_penalty * (deviation ** 2)

    def _resolve_identity(
        self,
        yolo_id: int,
        embedding,
        assigned: set,
        threshold: float,
        max_players: int,
        box: list,
        frame_width: int,
        ambiguity_threshold: float = 0.85,
    ):
        if yolo_id in self.id_mapping and self.id_mapping[yolo_id] not in assigned:
            return self.id_mapping[yolo_id]

        penalty = self._centrality_penalty(box, frame_width)
        matches = []
        for pid, data in self.player_gallery.items():
            if pid in assigned:
                continue
            mean_emb = np.mean(np.array(data["embeddings"][-20:]), axis=0).reshape(1, -1)
            dist = cdist(embedding, mean_emb, metric="cosine")[0][0]
            matches.append((dist + penalty, pid))

        if not matches:
            if len(self.player_gallery) < max_players and penalty < 0.1:
                return self._register_new_player(yolo_id, embedding)
            return None

        matches.sort(key=lambda x: x[0])
        best_dist, best_pid = matches[0]
        is_ambiguous = len(matches) > 1 and (best_dist / matches[1][0] > ambiguity_threshold)

        if best_dist < threshold and not is_ambiguous:
            self.id_mapping[yolo_id] = best_pid
            return best_pid
        if not is_ambiguous and len(self.player_gallery) < max_players and penalty < 0.1:
            return self._register_new_player(yolo_id, embedding)

        return None

    def _register_new_player(self, yolo_id: int, embedding) -> int:
        new_id = self.next_unique_id
        self.next_unique_id += 1
        self.id_mapping[yolo_id] = new_id
        self.player_gallery[new_id] = {"embeddings": [embedding], "last_seen": 0}
        return new_id

    def _update_best_init_frame(self, results, frame, frame_height: int, frame_id: int) -> None:
        if results[0].boxes.id is None:
            return
        detections = []
        for box_obj, yolo_id in zip(results[0].boxes, results[0].boxes.id.int().cpu().tolist()):
            box = box_obj.xyxy.cpu().tolist()[0]
            if self._is_near_side_player(box, frame_height):
                detections.append({"box": box, "yolo_id": yolo_id})
        if len(detections) > len(self.best_init_frame["detections"]):
            self.best_init_frame = {"frame": frame.copy(), "detections": detections}
            print(f"  Init candidate frame {frame_id}: {len(detections)} players")

    def _perform_initial_assignment(self, max_players: int, frame_id: int) -> None:
        data = self.best_init_frame
        if not data or not data["detections"]:
            print("No suitable frame for initial assignment — skipping")
            self.initial_assignment_done = True
            return

        frame, detections = data["frame"], data["detections"]
        detections.sort(key=lambda d: d["box"][3])  # sort by bottom-y
        assigned_id = 1

        for det in detections:
            if len(self.player_gallery) >= max_players:
                break
            embedding = self._get_embedding(frame, det["box"])
            if embedding is not None:
                self.player_gallery[assigned_id] = {"embeddings": [embedding], "last_seen": frame_id}
                if "yolo_id" in det:
                    self.id_mapping[det["yolo_id"]] = assigned_id
                assigned_id += 1

        if self.player_gallery:
            self.next_unique_id = assigned_id
            print(f"Initial assignment: {len(self.player_gallery)} players registered")
        self.initial_assignment_done = True

    def _prune_stale_ids(self, frame_id: int, stale_after: int) -> None:
        stale = [
            pid for pid, data in self.player_gallery.items()
            if frame_id - data["last_seen"] > stale_after
        ]
        for pid in stale:
            del self.player_gallery[pid]
            for y_id, p_id in list(self.id_mapping.items()):
                if p_id == pid:
                    del self.id_mapping[y_id]
        if stale:
            print(f"  Pruned {len(stale)} stale player IDs")
