"""nvarguscamerasrc-backed CameraSource for the IMX477.

ARGUS outputs NV12-family frames directly; nvvidconv+videoconvert convert
to a BGR numpy array for OpenCV/YOLO.

Whether the 1080p mode is a sensor crop or a downscale of the full
4032x3040 sensor is unconfirmed -- affects the true field of view.
"""

from __future__ import annotations

import logging

import cv2

logger = logging.getLogger(__name__)


def build_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1920,
    capture_height: int = 1080,
    framerate: int = 30,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={capture_width},height={capture_height},"
        f"framerate={framerate}/1,format=NV12 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1"
    )


class GStreamerCameraSource:
    def __init__(
        self,
        sensor_id: int = 0,
        capture_width: int = 1920,
        capture_height: int = 1080,
        framerate: int = 30,
    ):
        self.width = capture_width
        self.height = capture_height
        pipeline = build_pipeline(sensor_id, capture_width, capture_height, framerate)
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera pipeline: {pipeline}")
        logger.info("Opened camera: %s", pipeline)

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "GStreamerCameraSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
