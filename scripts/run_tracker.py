#!/usr/bin/env python3
"""Main entrypoint: real IMX477 feed + YOLO26 tracking + live pan/tilt
correction. Pass --rtsp to also serve the raw feed for remote viewing."""

import argparse
import logging
import sys

from pan_tilt_track.camera.gstreamer_source import GStreamerCameraSource
from pan_tilt_track.camera.rtsp_stream import RtspCameraServer
from pan_tilt_track.control.gain import ProportionalGain
from pan_tilt_track.control.loop import TrackingLoop
from pan_tilt_track.dynamixel import DynamixelConfig, PanTiltController
from pan_tilt_track.tracking.detector import YoloDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Verified against real hardware with scripts/sign_check.py: pan needs a
# NEGATIVE kp on this mount, tilt positive. Do not unify these into a
# shared constant -- the sign difference is mount-specific, not a
# copy-paste artifact.
DEADBAND_PX = 6.0
PAN_KP = -0.15
TILT_KP = 0.15


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", action="store_true", help="also serve the feed over RTSP")
    parser.add_argument("--rtsp-port", default="8554", help="RTSP server port (default 8554)")
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="burn detection boxes / crosshair / error vector into the RTSP feed (adds latency)",
    )
    parser.add_argument("--model", default=None, help="default: yolo26n.pt, or yolo26n-pose.pt for --target-mode head")
    parser.add_argument("--classes", type=int, nargs="*", default=None)
    parser.add_argument(
        "--target-mode",
        choices=["body", "head"],
        default="body",
        help="aim at the bbox center (body) or head keypoints (head, needs a pose model)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print per-frame pixel error / goal position"
    )
    args = parser.parse_args()

    model_path = args.model or ("yolo26n-pose.pt" if args.target_mode == "head" else "yolo26n.pt")

    rtsp_server = None
    if args.rtsp:
        rtsp_server = RtspCameraServer(port=args.rtsp_port)
        rtsp_server.start()

    def on_step(info: dict) -> None:
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

    config = DynamixelConfig()
    with PanTiltController(config) as controller, GStreamerCameraSource() as camera:
        controller.initialize()
        detector = YoloDetector(model_path=model_path, classes=args.classes)
        loop = TrackingLoop(
            camera=camera,
            detector=detector,
            controller=controller,
            pan_gain=ProportionalGain(kp=PAN_KP, deadband_px=DEADBAND_PX),
            tilt_gain=ProportionalGain(kp=TILT_KP, deadband_px=DEADBAND_PX),
            on_frame=rtsp_server.push_frame if rtsp_server is not None else None,
            on_step=on_step if args.verbose else None,
            target_mode=args.target_mode,
            draw_overlay=args.overlay,
        )
        try:
            loop.run()
        except KeyboardInterrupt:
            logger.info("Stopping")
        finally:
            controller.shutdown()
            if rtsp_server is not None:
                rtsp_server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
