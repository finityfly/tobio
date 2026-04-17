"""Lazy-loaded model singletons.

Models are instantiated on first use and cached for the lifetime of the process.
Using functools.lru_cache on module-level functions gives us a clean, thread-safe
singleton pattern without managing global state manually.
"""

from functools import lru_cache

from fastapi import HTTPException, status


def _load_or_503(name: str, factory):
    try:
        return factory()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{name} weights not found: {exc}",
        ) from exc


@lru_cache(maxsize=None)
def get_ball_tracker():
    from config import settings
    from services.ball_tracker import BallTracker
    return _load_or_503("BallTracker", lambda: BallTracker(settings.models["ball_tracker"].local_path))


@lru_cache(maxsize=None)
def get_court_tracker():
    from config import settings
    from services.court_tracker import CourtTracker
    return _load_or_503("CourtTracker", lambda: CourtTracker(settings.models["court_tracker"].local_path))


@lru_cache(maxsize=None)
def get_action_classifier():
    from config import settings
    from services.action_classifier import ActionClassifier
    return _load_or_503("ActionClassifier", lambda: ActionClassifier(settings.models["action_classifier"].local_path))


@lru_cache(maxsize=None)
def get_serve_recognizer():
    from config import settings
    from services.serve_recognizer import ServeRecognizer
    return _load_or_503("ServeRecognizer", lambda: ServeRecognizer(settings.models["serve_recognizer"].local_path))


@lru_cache(maxsize=None)
def get_player_tracker():
    from config import settings
    from services.player_tracker import PlayerTracker
    return _load_or_503("PlayerTracker", lambda: PlayerTracker(settings.models["player_tracker"].local_path))
