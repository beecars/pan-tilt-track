#!/usr/bin/env python3
"""Empirically verify the pan/tilt sign convention in run_tracker.py.

Run with a person/object visible to the camera, moving side to side (pan)
and up/down (tilt). Prints pixel error and goal position each frame.
Correct sign: |pixel_error| trends toward 0. Wrong sign: it grows and the
position marches toward its limit instead of centering the target.
"""

import argparse
import sys

from pan_tilt_track.camera.gstreamer_source import GStreamerCameraSource
from pan_tilt_track.control.gain import ProportionalGain
from pan_tilt_track.control.loop import TrackingLoop
from pan_tilt_track.dynamixel import DynamixelConfig, PanTiltController
from pan_tilt_track.tracking.detector import YoloDetector

DEADBAND_PX = 6.0
# Pan needs a negative kp on this mount -- positive drives it the wrong way.
PAN_KP = -0.15
TILT_KP = 0.15


def print_step(info: dict) -> None:
    if info["pixel_error_x"] is None:
        print(f"no target ({info.get('num_detections', 0)} detections this frame)")
    elif info["pan_position"] is None:
        print(f"err=({info['pixel_error_x']:+7.1f}, {info['pixel_error_y']:+7.1f})  (in deadband)")
    else:
        print(
            f"err=({info['pixel_error_x']:+7.1f}, {info['pixel_error_y']:+7.1f})  "
            f"delta=({info['pan_delta']:+5d}, {info['tilt_delta']:+5d})  "
            f"goal=({info['pan_position']:5d}, {info['tilt_position']:5d})"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-mode", choices=["body", "head"], default="body")
    parser.add_argument("--model", default=None, help="default: yolo26n.pt, or yolo26n-pose.pt for --target-mode head")
    args = parser.parse_args()
    model_path = args.model or ("yolo26n-pose.pt" if args.target_mode == "head" else "yolo26n.pt")

    config = DynamixelConfig()
    with PanTiltController(config) as controller, GStreamerCameraSource() as camera:
        controller.initialize()
        detector = YoloDetector(model_path=model_path, classes=[0])  # class 0 = person in COCO
        loop = TrackingLoop(
            camera=camera,
            detector=detector,
            controller=controller,
            pan_gain=ProportionalGain(kp=PAN_KP, deadband_px=DEADBAND_PX),
            tilt_gain=ProportionalGain(kp=TILT_KP, deadband_px=DEADBAND_PX),
            on_step=print_step,
            target_mode=args.target_mode,
        )
        try:
            loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            controller.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
