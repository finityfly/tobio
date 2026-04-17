import os
import warnings

from ultralytics import YOLO


class BaseTracker:
    """Shared base for all YOLO-backed trackers.

    Subclasses only need to define their inference methods — model loading
    and path validation are handled here.
    """

    def __init__(self, local_model_path: str) -> None:
        warnings.filterwarnings("ignore")
        self.local_model_path = local_model_path
        self.model: YOLO = self._load_model()

    def _load_model(self) -> YOLO:
        if not os.path.exists(self.local_model_path):
            raise FileNotFoundError(
                f"Model weights not found at '{self.local_model_path}'. "
                "Ensure the model file is present before starting the server."
            )
        return YOLO(self.local_model_path)
