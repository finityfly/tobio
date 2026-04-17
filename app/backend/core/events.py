from typing import List, Optional

from core.camera import Camera
from utils.geometry import calculate_iou

NET_HEIGHT_M = 2.43


def link_actions_to_players(
    action_detections: list,
    player_tracks: dict,
    ball_3d_positions: List[Optional[tuple]],
    camera: Camera,
    calibration_params: dict,
    iou_threshold: float = 0.2,
) -> list:
    """Enrich action detections with player IDs and ball/block metrics."""
    volleyball_events = []
    tracks_by_frame = {int(k): v for k, v in player_tracks.items()}

    for action in action_detections:
        frame = action.get("start_frame")
        action_box = action.get("box")

        if frame is None or action_box is None:
            volleyball_events.append(action)
            continue

        best_player, max_iou = None, 0.0
        for player in tracks_by_frame.get(frame, []):
            player_box = player.get("box")
            if player_box:
                iou = calculate_iou(action_box, player_box)
                if iou > max_iou:
                    max_iou = iou
                    best_player = player

        event = action.copy()

        if max_iou > iou_threshold and best_player is not None:
            event["player_id"] = best_player.get("player_id")
            action_name = action.get("action")

            ball_pos = _nearest_ball_pos(ball_3d_positions, frame)

            if ball_pos and action_name in ("spike", "set"):
                event["ball_height_m"] = round(ball_pos[2], 2)
                if action_name == "set":
                    event["set_position"] = round((-2 / 9) * ball_pos[0] + 4, 1)

            if action_name == "block" and camera.rvec is not None:
                player_box = best_player.get("box")
                top_center = ((player_box[0] + player_box[2]) / 2, player_box[1])
                block_pos = camera.get_point_3d_position(
                    top_center,
                    reference_real_height_m=NET_HEIGHT_M,
                    z_scale_calibration=calibration_params["z_scale"],
                    x_sensitivity=calibration_params["x_sens"],
                    ground_plane_offset=calibration_params["g_offset"],
                )
                if block_pos:
                    event["block_height_m"] = round(block_pos[2], 2)

        volleyball_events.append(event)

    return volleyball_events


def _nearest_ball_pos(
    ball_3d_positions: List[Optional[tuple]], frame: int
) -> Optional[tuple]:
    for offset in (0, -1, 1):
        idx = frame + offset
        if 0 <= idx < len(ball_3d_positions) and ball_3d_positions[idx] is not None:
            return ball_3d_positions[idx]
    return None
