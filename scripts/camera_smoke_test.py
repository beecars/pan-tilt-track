#!/usr/bin/env python3
"""Camera-only smoke test: confirm the nvarguscamerasrc pipeline works
before wiring in detection. Captures a few frames and saves the last one.

Sensor id/orientation come from config/cameras.json (see
pan_tilt_track.camera.config) -- physical mount properties, not runtime
flags. Edit that file if the cameras are swapped or remounted.
"""

import argparse
import sys

import cv2

from pan_tilt_track.camera.config import DEFAULT_CAMERA_CONFIG_PATH, load_cameras_config
from pan_tilt_track.camera.gstreamer_source import GStreamerCameraSource


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--out", default="camera_smoke_test.jpg")
    parser.add_argument("--camera-config", default=DEFAULT_CAMERA_CONFIG_PATH)
    parser.add_argument("--role", choices=["wide", "telephoto"], default="telephoto")
    args = parser.parse_args()

    cameras = load_cameras_config(args.camera_config)
    cam_cfg = getattr(cameras, args.role)

    with GStreamerCameraSource(
        sensor_id=cam_cfg.sensor_id,
        capture_width=cam_cfg.capture_width,
        capture_height=cam_cfg.capture_height,
        framerate=cam_cfg.framerate,
        flip_method=cam_cfg.flip_method,
    ) as camera:
        frame = None
        for i in range(args.frames):
            frame = camera.read()
            if frame is None:
                print(f"frame {i}: no frame received", file=sys.stderr)
                return 1
        cv2.imwrite(args.out, frame)
        print(f"Captured {args.frames} frames, saved last to {args.out} ({frame.shape})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
