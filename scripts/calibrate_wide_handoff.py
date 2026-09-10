#!/usr/bin/env python3
"""Calibrates the wide-camera-pixel -> pan/tilt-goal-tick mapping used by
run_tracker.py's wide-driven acquisition stage.

Run with a person walking around, visible to both cameras at least part
of the time. Uses telephoto's self-tracking loop to log (wide-frame pixel
offset, current pan/tilt ticks) samples whenever telephoto is
well-centered on its target and wide sees exactly one confident
detection. Collects until Ctrl+C or --min-samples, fits a linear model
per axis, and writes config/wide_handoff.json.

Re-run after any reassembly or camera remount.
"""

import argparse
import logging
import sys

import numpy as np

from pan_tilt_track.camera.config import DEFAULT_CAMERA_CONFIG_PATH, load_cameras_config
from pan_tilt_track.camera.gstreamer_source import GStreamerCameraSource
from pan_tilt_track.control.gain import ProportionalGain
from pan_tilt_track.control.gains import DEADBAND_PX, PAN_KP, TILT_KP
from pan_tilt_track.control.loop import TrackingLoop
from pan_tilt_track.control.wide_handoff import (
    DEFAULT_WIDE_HANDOFF_CONFIG_PATH,
    WideHandoffCalibration,
    save_wide_handoff_config,
)
from pan_tilt_track.dynamixel import DEFAULT_SERVO_CONFIG_PATH, PanTiltController, load_dynamixel_config
from pan_tilt_track.tracking.detector import YoloDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Minimum fraction of the wide frame's width/height that collected samples
# should span before the fit is trustworthy: a cluster of samples near
# frame center under-constrains the slope.
MIN_SPREAD_FRACTION = 0.3


def fit_axis(offsets: list[float], ticks: list[float], frame_extent: float, axis_name: str) -> tuple[float, float]:
    spread = max(offsets) - min(offsets) if offsets else 0.0
    if spread < MIN_SPREAD_FRACTION * frame_extent:
        print(
            f"WARNING: {axis_name} samples only span {spread:.0f}px of a {frame_extent:.0f}px "
            f"frame -- walk further to the extremes for a better-constrained fit.",
            file=sys.stderr,
        )
    slope, intercept = np.polyfit(offsets, ticks, 1)
    return float(intercept), float(slope)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-mode", choices=["body", "head"], default="head")
    parser.add_argument("--model", default=None, help="default: yolo26n.pt, or yolo26n-pose.pt for --target-mode head")
    parser.add_argument("--camera-config", default=DEFAULT_CAMERA_CONFIG_PATH)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG_PATH)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--out", default=DEFAULT_WIDE_HANDOFF_CONFIG_PATH)
    args = parser.parse_args()
    model_path = args.model or ("yolo26n-pose.pt" if args.target_mode == "head" else "yolo26n.pt")

    cameras = load_cameras_config(args.camera_config)
    wide, telephoto = cameras.wide, cameras.telephoto
    config = load_dynamixel_config(args.servo_config)

    pan_samples: list[tuple[float, float]] = []  # (dx, pan_tick)
    tilt_samples: list[tuple[float, float]] = []  # (dy, tilt_tick)

    with PanTiltController(config) as controller, GStreamerCameraSource(
        sensor_id=telephoto.sensor_id,
        capture_width=telephoto.capture_width,
        capture_height=telephoto.capture_height,
        framerate=telephoto.framerate,
        flip_method=telephoto.flip_method,
    ) as telephoto_camera, GStreamerCameraSource(
        sensor_id=wide.sensor_id,
        capture_width=wide.capture_width,
        capture_height=wide.capture_height,
        framerate=wide.framerate,
        flip_method=wide.flip_method,
    ) as wide_camera:
        controller.initialize()
        wide_detector = YoloDetector(model_path=model_path, classes=[0])  # class 0 = person in COCO

        def on_step(info: dict) -> None:
            if info["pixel_error_x"] is None or info.get("pan_delta") != 0 or info.get("tilt_delta") != 0:
                return  # no target, or mid-correction: not well-centered yet
            wide_frame = wide_camera.read()
            if wide_frame is None:
                return
            detections = wide_detector.track(wide_frame)
            if len(detections) != 1:
                return
            wx, wy = detections[0].target_point(args.target_mode)
            wide_h, wide_w = wide_frame.shape[:2]
            dx, dy = wx - wide_w / 2, wy - wide_h / 2
            pan_tick = controller.read_present_position(config.pan_id)
            tilt_tick = controller.read_present_position(config.tilt_id)
            pan_samples.append((dx, pan_tick))
            tilt_samples.append((dy, tilt_tick))
            if len(pan_samples) % 10 == 0:
                print(f"collected {len(pan_samples)} samples...")

        telephoto_detector = YoloDetector(model_path=model_path, classes=[0])
        loop = TrackingLoop(
            camera=telephoto_camera,
            detector=telephoto_detector,
            controller=controller,
            pan_gain=ProportionalGain(kp=PAN_KP, deadband_px=DEADBAND_PX),
            tilt_gain=ProportionalGain(kp=TILT_KP, deadband_px=DEADBAND_PX),
            on_step=on_step,
            target_mode=args.target_mode,
        )
        try:
            while len(pan_samples) < args.min_samples:
                if not loop.step():
                    break
        except KeyboardInterrupt:
            pass
        finally:
            controller.shutdown()

    if len(pan_samples) < 2:
        print(f"Only {len(pan_samples)} sample(s) collected -- not enough to fit. Aborting.", file=sys.stderr)
        return 1

    print(f"Fitting from {len(pan_samples)} samples.")
    pan_dx, pan_ticks = zip(*pan_samples)
    tilt_dy, tilt_ticks = zip(*tilt_samples)
    pan_tick_at_center, pan_ticks_per_px = fit_axis(list(pan_dx), list(pan_ticks), wide.capture_width, "pan")
    tilt_tick_at_center, tilt_ticks_per_px = fit_axis(list(tilt_dy), list(tilt_ticks), wide.capture_height, "tilt")

    calibration = WideHandoffCalibration(
        pan_tick_at_center=pan_tick_at_center,
        pan_ticks_per_px=pan_ticks_per_px,
        tilt_tick_at_center=tilt_tick_at_center,
        tilt_ticks_per_px=tilt_ticks_per_px,
    )
    save_wide_handoff_config(calibration, args.out)
    print(f"Wrote {args.out}: {calibration}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
