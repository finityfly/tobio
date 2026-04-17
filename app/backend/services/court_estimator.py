import cv2
import numpy as np


class CourtEstimator:
    def __init__(self, court_class_id: int = 1, approx_epsilon_factor: float = 0.02):
        self.court_class_id = court_class_id
        self.epsilon_factor = approx_epsilon_factor

    def predict(self, results) -> dict:
        if results.masks is None:
            return self._empty("not_found")

        raw_contours, confidences = [], []
        for i, cls_tensor in enumerate(results.boxes.cls):
            if int(cls_tensor.item()) == self.court_class_id:
                raw_contours.append(results.masks.xy[i].astype(np.int32))
                confidences.append(results.boxes.conf[i].item())

        if not raw_contours:
            return self._empty("not_found")

        all_points = np.vstack(raw_contours)
        hull = cv2.convexHull(all_points)
        epsilon = self.epsilon_factor * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)
        corners = self._reduce_to_4_points(approx)

        return {
            "corners": corners.tolist(),
            "confidence": round(max(confidences), 4),
            "status": "success" if len(corners) == 4 else "partial",
        }

    def _empty(self, status: str) -> dict:
        return {"corners": [], "confidence": 0.0, "status": status}

    def _reduce_to_4_points(self, points: np.ndarray) -> np.ndarray:
        poly = [tuple(p) for p in points]
        if len(poly) <= 4:
            return np.array(poly, dtype=np.int32)

        while len(poly) > 4:
            n = len(poly)
            # find the shortest edge
            min_d, min_idx = float("inf"), -1
            for i in range(n):
                p1, p2 = poly[i], poly[(i + 1) % n]
                d = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                if d < min_d:
                    min_d, min_idx = d, i

            idx = min_idx
            p_prev = poly[(idx - 1 + n) % n]
            p_curr = poly[idx]
            p_next = poly[(idx + 1) % n]
            p_fut = poly[(idx + 2) % n]

            intersection = self._line_intersection(p_prev, p_curr, p_next, p_fut)
            if intersection is not None:
                if (idx + 1) % n == 0:
                    poly.pop(0)
                    poly.pop(-1)
                    poly.append(intersection)
                else:
                    poly.pop(idx + 1)
                    poly.pop(idx)
                    poly.insert(idx, intersection)
            else:
                poly.pop(idx)

        return np.array(poly, dtype=np.int32)

    @staticmethod
    def _line_intersection(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if den == 0:
            return None
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
        return int(x1 + ua * (x2 - x1)), int(y1 + ua * (y2 - y1))
