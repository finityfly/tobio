from typing import List, Optional, Tuple

import numpy as np


def calculate_iou(box_a: list, box_b: list) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    if inter_area == 0:
        return 0.0
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter_area / float(box_a_area + box_b_area - inter_area)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL: smallest sum
    rect[2] = pts[np.argmax(s)]   # BR: largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR: smallest diff
    rect[3] = pts[np.argmax(diff)]  # BL: largest diff
    return rect


def moving_average(
    data: List[Optional[Tuple]], window_size: int = 5
) -> List[Optional[Tuple]]:
    if not data:
        return []
    smoothed = []
    for i in range(len(data)):
        window = [p for p in data[max(0, i - window_size + 1): i + 1] if p is not None]
        if not window:
            smoothed.append(None)
        else:
            smoothed.append((
                float(np.mean([p[0] for p in window])),
                float(np.mean([p[1] for p in window])),
                float(np.mean([p[2] for p in window])),
            ))
    return smoothed
