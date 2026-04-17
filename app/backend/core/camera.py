import cv2
import numpy as np


class Camera:
    # Volleyball court dimensions in meters (X=width, Z=depth, Y=height)
    COURT_WORLD_COORDS = np.array([
        [0, 0, 0],   # TL
        [9, 0, 0],   # TR
        [9, 0, 18],  # BR
        [0, 0, 18],  # BL
    ], dtype=np.float32)

    def __init__(self, camera_height_m: float = None):
        self.rvec = None
        self.tvec = None
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        self.camera_height_m = camera_height_m
        self.world_coords = self.COURT_WORLD_COORDS.copy()

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order 4 corner points as: top-left, top-right, bottom-right, bottom-left."""
        y_sorted = pts[np.argsort(pts[:, 1]), :]
        top = y_sorted[:2][np.argsort(y_sorted[:2, 0])]
        bottom = y_sorted[2:][np.argsort(y_sorted[2:, 0])]
        tl, tr = top
        bl, br = bottom
        return np.array([tl, tr, br, bl], dtype="float32")

    def calibrate(
        self,
        image_corners: np.ndarray,
        image_width: int = 1920,
        image_height: int = 1080,
        focal_length_multiplier: float = 1.2,
    ) -> bool:
        if image_corners is None or len(image_corners) != 4:
            return False

        focal_length = image_width * focal_length_multiplier
        self.camera_matrix = np.array(
            [[focal_length, 0, image_width / 2],
             [0, focal_length, image_height / 2],
             [0, 0, 1]],
            dtype=np.float32,
        )

        initial_rvec = np.array([np.pi, 0, 0], dtype=np.float32)
        initial_tvec = np.array([4.5, self.camera_height_m or 10.0, 9.0], dtype=np.float32)

        success, self.rvec, self.tvec = cv2.solvePnP(
            self.world_coords,
            image_corners,
            self.camera_matrix,
            self.dist_coeffs,
            rvec=initial_rvec,
            tvec=initial_tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return False

        if self.camera_height_m and self.camera_height_m > 0:
            R, _ = cv2.Rodrigues(self.rvec)
            cam_pos = -np.dot(np.linalg.inv(R), self.tvec)
            target_pos = np.array(
                [cam_pos[0], self.camera_height_m, cam_pos[2]]
            ).reshape(3, 1)
            self.tvec = -np.dot(R, target_pos)

        return True

    def get_3d_position_estimation(
        self,
        bbox: list,
        ball_real_diameter_m: float = 0.21,
        z_scale_calibration: float = 1.5,
        x_sensitivity: float = 2.0,
        ground_plane_offset: float = 0.0,
    ) -> tuple:
        fallback = (4.5, 9.0, 1.0)
        if self.rvec is None or self.camera_matrix is None or bbox is None:
            return fallback

        x_min, y_min, x_max, y_max = bbox
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        pixel_diameter = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        if pixel_diameter < 5:
            return fallback

        try:
            focal_x = self.camera_matrix[0, 0]
            depth = (focal_x * ball_real_diameter_m) / pixel_diameter

            undistorted = cv2.undistortPoints(
                np.array([[[cx, cy]]], dtype=np.float32),
                self.camera_matrix,
                self.dist_coeffs,
            )
            px, py = undistorted[0][0]

            ball_cam = np.array([px * depth, py * depth, depth]).reshape(3, 1)
            R, _ = cv2.Rodrigues(self.rvec)
            ball_world = np.dot(np.linalg.inv(R), (ball_cam - self.tvec)).flatten()

            w_x = ball_world[0]
            w_y = ball_world[2]
            raw_h = ball_world[1]

            w_x = 4.5 + (w_x - 4.5) * x_sensitivity
            w_z = (raw_h - ground_plane_offset) * z_scale_calibration

            return (
                float(max(-5.0, min(14.0, w_x))),
                float(max(-5.0, min(23.0, w_y))),
                float(max(0.0, min(15.0, w_z))),
            )
        except Exception:
            return fallback

    def get_point_3d_position(
        self,
        point_2d: tuple,
        reference_real_height_m: float,
        z_scale_calibration: float = 1.5,
        x_sensitivity: float = 2.0,
        ground_plane_offset: float = 0.0,
    ) -> tuple:
        fallback = (4.5, 9.0, reference_real_height_m)
        if self.rvec is None or self.camera_matrix is None or point_2d is None:
            return fallback

        try:
            undistorted = cv2.undistortPoints(
                np.array([[[point_2d[0], point_2d[1]]]], dtype=np.float32),
                self.camera_matrix,
                self.dist_coeffs,
            )
            px, py = undistorted[0][0]

            R, _ = cv2.Rodrigues(self.rvec)
            R_inv = np.linalg.inv(R)
            cam_pos = -np.dot(R_inv, self.tvec).flatten()
            ray_dir = np.dot(R_inv, np.array([px, py, 1.0]))

            if abs(ray_dir[1]) < 1e-6:
                return fallback

            t = (reference_real_height_m - cam_pos[1]) / ray_dir[1]
            pos = (cam_pos + t * ray_dir).flatten()

            w_x = 4.5 + (pos[0] - 4.5) * x_sensitivity
            w_z = (pos[1] - ground_plane_offset) * z_scale_calibration

            return (
                float(max(-5.0, min(14.0, w_x))),
                float(max(-5.0, min(23.0, pos[2]))),
                float(max(0.0, min(15.0, w_z))),
            )
        except Exception:
            return fallback
