#!/usr/bin/env python3
"""Dual-camera smoke test: confirm both IMX477 sensors can be captured
concurrently via nvarguscamerasrc before wiring in detection/tracking.
Captures a few frames from each and saves a side-by-side comparison image.

Per-sensor id/orientation come from config/cameras.json (see
pan_tilt_track.camera.config) -- physical mount properties, not runtime
flags. Edit that file if the cameras are swapped or remounted.
"""

import argparse
import sys

import cv2
import numpy as np

from pan_tilt_track.camera.config import DEFAULT_CAMERA_CONFIG_PATH, load_cameras_config
from pan_tilt_track.camera.gstreamer_source import GStreamerCameraSource


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--out", default="dual_camera_smoke_test.jpg")
    parser.add_argument("--camera-config", default=DEFAULT_CAMERA_CONFIG_PATH)
    args = parser.parse_args()

    cameras = load_cameras_config(args.camera_config)
    wide, telephoto = cameras.wide, cameras.telephoto

    with GStreamerCameraSource(
        sensor_id=wide.sensor_id,
        capture_width=wide.capture_width,
        capture_height=wide.capture_height,
        framerate=wide.framerate,
        flip_method=wide.flip_method,
    ) as wide_cam, GStreamerCameraSource(
        sensor_id=telephoto.sensor_id,
        capture_width=telephoto.capture_width,
        capture_height=telephoto.capture_height,
        framerate=telephoto.framerate,
        flip_method=telephoto.flip_method,
    ) as telephoto_cam:
        wide_frame = telephoto_frame = None
        for i in range(args.frames):
            wide_frame = wide_cam.read()
            telephoto_frame = telephoto_cam.read()
            if wide_frame is None:
                print(f"frame {i}: no frame received from wide camera", file=sys.stderr)
                return 1
            if telephoto_frame is None:
                print(f"frame {i}: no frame received from telephoto camera", file=sys.stderr)
                return 1

        combined = np.hstack((wide_frame, telephoto_frame))
        cv2.imwrite(args.out, combined)
        print(
            f"Captured {args.frames} frames from both sensors, "
            f"saved side-by-side to {args.out} "
            f"(wide={wide_frame.shape}, telephoto={telephoto_frame.shape})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
