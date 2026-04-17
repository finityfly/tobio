import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModelConfig:
    s3_bucket: str
    s3_key: str
    local_path: str


@dataclass
class Settings:
    # Auth (read from environment; defaults are for local dev only)
    api_username: str = field(default_factory=lambda: os.getenv("API_USERNAME", "tobio"))
    api_password: str = field(default_factory=lambda: os.getenv("API_PASSWORD", "tobio"))

    # Runtime paths
    cache_dir: str = "cache"
    demo_court_lines_path: str = "cache/demo_vod_court_lines.json"

    # CORS
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://tobio.daniellu.ca",
    ])

    # ML model configurations
    models: Dict[str, ModelConfig] = field(default_factory=lambda: {
        "court_tracker": ModelConfig(
            s3_bucket="tobio-models",
            s3_key="yolov11-court-seg-v2/weights/best.pt",
            local_path="models/best_courttracker.pt",
        ),
        "ball_tracker": ModelConfig(
            s3_bucket="tobio-models",
            s3_key="yolov11-volleyball-v2/weights/best.pt",
            local_path="models/best_balltracker.pt",
        ),
        "action_classifier": ModelConfig(
            s3_bucket="tobio-models",
            s3_key="yolov11-actions-v2/weights/best.pt",
            local_path="models/best_actionclassifier.pt",
        ),
        "serve_recognizer": ModelConfig(
            s3_bucket="tobio-models",
            s3_key="yolov11-serve/weights/best.pt",
            local_path="models/best_serverecognizer.pt",
        ),
        "player_tracker": ModelConfig(
            s3_bucket="tobio-models",
            s3_key="yolov11-player-fixed/weights/best.pt",
            local_path="models/best_playertracker.pt",
        ),
    })


settings = Settings()
