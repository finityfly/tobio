import json
import os

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile

from config import settings
from core.camera import Camera
from core.events import link_actions_to_players
from core.model_registry import (
    get_action_classifier,
    get_ball_tracker,
    get_court_tracker,
    get_player_tracker,
    get_serve_recognizer,
)
from core.security import verify_credentials
from utils.cache import load_from_cache, save_to_cache
from utils.video import save_temp_video

router = APIRouter(tags=["video"])

_DEFAULT_ACTION_COOLDOWNS = {
    "serve": 120,
    "spike": 45,
    "block": 45,
    "set": 30,
    "defense": 30,
}


@router.post("/process-court-lines")
@router.post("/process-court-lines/")
def process_court_lines(
    file: UploadFile = File(...),
    username: str = Depends(verify_credentials),
):
    # Serve cached result for the demo video
    if file.filename and file.filename.lower() == "demo_vod.mp4":
        if os.path.exists(settings.demo_court_lines_path):
            with open(settings.demo_court_lines_path) as f:
                return json.load(f)

    temp_path = save_temp_video(file, settings.cache_dir)
    try:
        result = get_court_tracker().track_court(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result


@router.post("/process-video")
@router.post("/process-video/")
def process_video(
    file: UploadFile = File(...),
    court_corners: str = Form(None),
    camera_height: float = Form(7.0),
    focal_length: float = Form(2.0),
    ball_height_calibration: float = Form(1.0),
    ball_side_calibration: float = Form(2.0),
    ground_plane_offset: float = Form(0.0),
    username: str = Depends(verify_credentials),
):
    temp_path = save_temp_video(file, settings.cache_dir)
    base_name, _ = os.path.splitext(file.filename)
    cache_path = os.path.join(settings.cache_dir, f"{base_name}.json")

    calibration_params = {
        "z_scale": ball_height_calibration,
        "x_sens": ball_side_calibration,
        "g_offset": ground_plane_offset,
    }

    user_corners = json.loads(court_corners) if court_corners else None
    camera = Camera(camera_height_m=camera_height)

    # Calibrate camera if corners are provided
    if user_corners:
        video_width, video_height = _probe_video_dimensions(temp_path, cache_path)
        ordered = camera.order_points(np.array(user_corners, dtype=np.float32))
        camera.calibrate(ordered, video_width, video_height, focal_length)

    # --- Cache check ---
    cached = load_from_cache(cache_path)
    if not cached:
        print("--- CACHE MISS: running all trackers ---")
        cached = _run_all_trackers(temp_path, user_corners, camera, calibration_params)
        save_to_cache(cache_path, cached)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Re-compute 3D positions and events from (potentially re-calibrated) camera
    video_metadata = cached.get("video_metadata", {})
    ball_tracks = cached.get("ball_data", {}).get("ball_tracks", [])
    action_detections = cached.get("action_classifications", {}).get("action_detections", [])
    player_tracks = cached.get("player_data", {}).get("player_tracks", {})
    serve_events = cached.get("serve_data", {}).get("serve_events", [])
    total_frames = video_metadata.get("total_frames", 0)

    ball_detections = _build_ball_detections(ball_tracks, total_frames)
    ball_3d_positions = _compute_ball_3d(ball_detections, total_frames, camera, calibration_params)
    volleyball_events = link_actions_to_players(
        action_detections, player_tracks, ball_3d_positions, camera, calibration_params
    )

    return {
        "video_metadata": video_metadata,
        "court_detections": [user_corners] * total_frames if user_corners else [],
        "ball_detections": ball_detections,
        "action_detections": action_detections,
        "serve_events": serve_events,
        "player_tracks": player_tracks,
        "ball_3d_positions": ball_3d_positions,
        "volleyball_events": volleyball_events,
    }


@router.post("/track-ball")
@router.post("/track-ball/")
def track_ball(
    file: UploadFile = File(...),
    username: str = Depends(verify_credentials),
):
    temp_path = save_temp_video(file, settings.cache_dir)
    try:
        result = get_ball_tracker().track_ball(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return result


@router.post("/track-court")
@router.post("/track-court/")
def track_court(
    file: UploadFile = File(...),
    username: str = Depends(verify_credentials),
):
    temp_path = save_temp_video(file, settings.cache_dir)
    try:
        result = get_court_tracker().track_court(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return result


# ------------------------------------------------------------------ #
# Private helpers                                                      #
# ------------------------------------------------------------------ #

def _probe_video_dimensions(video_path: str, cache_path: str) -> tuple[int, int]:
    """Return (width, height) — from cache metadata if available, else via OpenCV."""
    cached = load_from_cache(cache_path)
    if cached and "video_metadata" in cached:
        meta = cached["video_metadata"]
        return meta.get("width", 1920), meta.get("height", 1080)
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def _run_all_trackers(
    video_path: str, user_corners: list, camera: Camera, calibration_params: dict
) -> dict:
    ball_data = get_ball_tracker().track_ball(video_path, conf_thresh=0.15)
    action_data = get_action_classifier().classify_action(
        video_path,
        conf_thresh=0.2,
        sliding_window_size=3,
        action_cooldowns=_DEFAULT_ACTION_COOLDOWNS,
        trigger_count=1,
    )
    player_data = get_player_tracker().track_players(
        video_path, conf_thresh=0.3, reid_sim_threshold=0.2
    )
    court_data = get_court_tracker().track_court(video_path, conf_thresh=0.3)
    serve_data = get_serve_recognizer().recognize_serves(
        video_path, court_corners=user_corners, conf_thresh=0.6, cooldown_frames=120
    )

    total_frames = player_data.get("video_metadata", {}).get("total_frames", 0)
    ball_detections = _build_ball_detections(ball_data.get("ball_tracks", []), total_frames)
    ball_3d = _compute_ball_3d(ball_detections, total_frames, camera, calibration_params)
    volleyball_events = link_actions_to_players(
        action_data.get("action_detections", []),
        player_data.get("player_tracks", {}),
        ball_3d,
        camera,
        calibration_params,
    )

    return {
        "video_metadata": player_data.get("video_metadata", {}),
        "ball_data": ball_data,
        "action_classifications": action_data,
        "player_data": player_data,
        "serve_data": serve_data,
        "court_data": court_data,
        "volleyball_events": volleyball_events,
    }


def _build_ball_detections(ball_tracks: list, total_frames: int) -> list:
    detections = [None] * total_frames
    for track in ball_tracks:
        if track["frame"] < total_frames:
            detections[track["frame"]] = track["bbox"]
    return detections


def _compute_ball_3d(
    ball_detections: list, total_frames: int, camera: Camera, calibration_params: dict
) -> list:
    positions = [None] * total_frames
    if camera.rvec is None:
        return positions
    for i, bbox in enumerate(ball_detections):
        if bbox:
            positions[i] = camera.get_3d_position_estimation(
                bbox,
                z_scale_calibration=calibration_params["z_scale"],
                x_sensitivity=calibration_params["x_sens"],
                ground_plane_offset=calibration_params["g_offset"],
            )
    return positions
