"""Camera interface for ARX X5 robot setup.

Handles three fish-eye cameras (left, right, base) for visual observations.
Processes images to match π0.5 VLA input format (224x224 RGB).

Author: Billy Chern (Shichen)
License: MIT
"""

import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class CameraConfig:
    """Configuration for a single camera."""

    def __init__(
        self,
        device_id: int,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        output_size: Tuple[int, int] = (224, 224),
    ):
        """Initialize camera configuration.

        Args:
            device_id: Camera device ID (e.g., 0, 1, 2)
            resolution: Camera capture resolution
            fps: Camera frame rate
            output_size: Output image size for VLA (default: 224x224)
        """
        self.device_id = device_id
        self.resolution = resolution
        self.fps = fps
        self.output_size = output_size


class TriCameraSystem:
    """Three-camera system for robot visual observations.

    Manages three fish-eye cameras positioned around the workspace:
    - Left camera: Left side view
    - Right camera: Right side view
    - Base camera: Top-down or front view

    Provides synchronized image capture and preprocessing for VLA input.

    Example:
        >>> camera_system = TriCameraSystem(
        ...     left_id=0, right_id=1, base_id=2
        ... )
        >>> images = camera_system.get_images()
        >>> # images = {'left': [224,224,3], 'right': [224,224,3], 'base': [224,224,3]}
    """

    def __init__(
        self,
        left_id: int = 0,
        right_id: int = 1,
        base_id: int = 2,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        output_size: Tuple[int, int] = (224, 224),
        fish_eye_correction: bool = False,
        verbose: bool = True,
    ):
        """Initialize tri-camera system.

        Args:
            left_id: Device ID for left camera
            right_id: Device ID for right camera
            base_id: Device ID for base camera
            resolution: Camera capture resolution
            fps: Camera frame rate
            output_size: Output image size (224x224 for π0.5)
            fish_eye_correction: Apply distortion correction for fish-eye lens
            verbose: Print initialization messages
        """
        self.output_size = output_size
        self.fish_eye_correction = fish_eye_correction
        self.verbose = verbose

        # Create camera configs
        self.configs = {
            "left": CameraConfig(left_id, resolution, fps, output_size),
            "right": CameraConfig(right_id, resolution, fps, output_size),
            "base": CameraConfig(base_id, resolution, fps, output_size),
        }

        # Initialize cameras
        self.cameras: Dict[str, cv2.VideoCapture] = {}
        self._initialize_cameras()

        # Distortion correction parameters (if using fish-eye correction)
        self.camera_matrices: Dict[str, Optional[np.ndarray]] = {
            "left": None,
            "right": None,
            "base": None,
        }
        self.dist_coeffs: Dict[str, Optional[np.ndarray]] = {
            "left": None,
            "right": None,
            "base": None,
        }

        if self.verbose:
            print("✓ Tri-camera system initialized successfully")

    def _initialize_cameras(self) -> None:
        """Initialize all camera connections."""
        for name, config in self.configs.items():
            if self.verbose:
                print(f"Initializing {name} camera (ID: {config.device_id})...")

            cap = cv2.VideoCapture(config.device_id)

            if not cap.isOpened():
                raise RuntimeError(f"Failed to open {name} camera (ID: {config.device_id})")

            # Set resolution and FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, config.fps)

            self.cameras[name] = cap

            if self.verbose:
                actual_res = (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  ✓ {name} camera: {actual_res} @ {actual_fps:.1f} FPS")

    def get_images(self) -> Dict[str, np.ndarray]:
        """Capture synchronized images from all cameras.

        Returns:
            Dictionary with keys 'left', 'right', 'base', each containing
            RGB image arrays of shape [224, 224, 3] with values in [0, 255] uint8.
        """
        images = {}

        for name, cap in self.cameras.items():
            ret, frame = cap.read()

            if not ret:
                raise RuntimeError(f"Failed to capture from {name} camera")

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply fish-eye correction if enabled
            if self.fish_eye_correction and self.camera_matrices[name] is not None:
                frame_rgb = self._undistort(frame_rgb, name)

            # Resize to output size
            frame_resized = cv2.resize(
                frame_rgb, self.output_size, interpolation=cv2.INTER_LINEAR
            )

            images[name] = frame_resized

        return images

    def _undistort(self, image: np.ndarray, camera_name: str) -> np.ndarray:
        """Apply distortion correction for fish-eye lens.

        Args:
            image: Input image
            camera_name: Name of camera ('left', 'right', 'base')

        Returns:
            Undistorted image
        """
        camera_matrix = self.camera_matrices[camera_name]
        dist_coeffs = self.dist_coeffs[camera_name]

        if camera_matrix is None or dist_coeffs is None:
            return image

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )

        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Crop to ROI
        x, y, w, h = roi
        undistorted = undistorted[y : y + h, x : x + w]

        return undistorted

    def calibrate_camera(
        self,
        camera_name: str,
        calibration_images: list,
        checkerboard_size: Tuple[int, int] = (9, 6),
    ) -> None:
        """Calibrate camera for fish-eye distortion correction.

        Args:
            camera_name: Name of camera to calibrate ('left', 'right', 'base')
            calibration_images: List of calibration images with checkerboard pattern
            checkerboard_size: Size of checkerboard (width, height in inner corners)
        """
        # Prepare object points
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0 : checkerboard_size[0], 0 : checkerboard_size[1]
        ].T.reshape(-1, 2)

        obj_points = []  # 3D points in real world space
        img_points = []  # 2D points in image plane

        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

            if ret:
                obj_points.append(objp)
                img_points.append(corners)

        if len(obj_points) == 0:
            raise ValueError("No checkerboard patterns found in calibration images")

        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )

        self.camera_matrices[camera_name] = camera_matrix
        self.dist_coeffs[camera_name] = dist_coeffs

        if self.verbose:
            print(f"✓ {camera_name} camera calibrated (reprojection error: {ret:.4f})")

    def close(self) -> None:
        """Release all camera connections."""
        if self.verbose:
            print("Closing cameras...")

        for name, cap in self.cameras.items():
            cap.release()
            if self.verbose:
                print(f"  ✓ {name} camera released")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def test_capture(self, duration: float = 5.0, display: bool = False) -> None:
        """Test camera capture for a duration.

        Args:
            duration: Test duration in seconds
            display: Display images using OpenCV window (requires GUI)
        """
        print(f"Testing camera capture for {duration}s...")
        start_time = time.time()
        frame_count = 0

        try:
            while time.time() - start_time < duration:
                images = self.get_images()
                frame_count += 1

                if display:
                    # Concatenate images horizontally for display
                    combined = np.concatenate(
                        [images["left"], images["base"], images["right"]], axis=1
                    )
                    # Convert RGB back to BGR for OpenCV display
                    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Tri-Camera System", combined_bgr)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                time.sleep(0.033)  # ~30 FPS

        finally:
            if display:
                cv2.destroyAllWindows()

        fps = frame_count / duration
        print(f"✓ Captured {frame_count} frames ({fps:.1f} FPS)")
