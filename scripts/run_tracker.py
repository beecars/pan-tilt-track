#!/usr/bin/env python3
"""Main entrypoint: two-stage wide-acquisition / telephoto-fine-tracking
pan/tilt correction. Pass --rtsp to also serve the feed for remote
viewing. See pan_tilt_track.control.wide_handoff and TrackingLoop for the
two stages."""

import argparse
import logging
import sys
from contextlib import ExitStack

from pan_tilt_track.camera.config import DEFAULT_CAMERA_CONFIG_PATH, load_cameras_config
from pan_tilt_track.camera.gstreamer_source import GStreamerCameraSource
from pan_tilt_track.stream.pip_relay import PipRtspRelay
from pan_tilt_track.stream.rtsp_stream import RtspCameraServer
from pan_tilt_track.control.gain import ProportionalGain
from pan_tilt_track.control.gains import DEADBAND_PX, PAN_KP, TILT_KP
from pan_tilt_track.control.loop import TrackingLoop
from pan_tilt_track.control.wide_handoff import (
    DEFAULT_WIDE_HANDOFF_CONFIG_PATH,
    WideHandoffMapper,
    load_wide_handoff_config,
)
from pan_tilt_track.dynamixel import DEFAULT_SERVO_CONFIG_PATH, PanTiltController, load_dynamixel_config
from pan_tilt_track.dynamixel.config import clamp_position
from pan_tilt_track.tracking.detector import ANIMAL_CLASS_IDS, YoloDetector
from pan_tilt_track.tracking.overlay import draw_mode_badge
from pan_tilt_track.tracking.track_manager import TrackManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtsp", action="store_true", help="also serve the feed over RTSP")
    parser.add_argument("--rtsp-port", default="8554", help="RTSP server port (default 8554)")
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="burn detection boxes / crosshair / error vector into the RTSP feed (adds latency)",
    )
    parser.add_argument("--camera-config", default=DEFAULT_CAMERA_CONFIG_PATH)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG_PATH)
    parser.add_argument("--wide-handoff-config", default=DEFAULT_WIDE_HANDOFF_CONFIG_PATH)
    parser.add_argument(
        "--rtsp-pip",
        action="store_true",
        help="composite the wide camera as a picture-in-picture inset (bottom-left) into the RTSP feed; requires --rtsp",
    )
    parser.add_argument("--pip-scale", type=float, default=0.25, help="inset width as a fraction of main frame width")
    parser.add_argument("--pip-margin", type=int, default=16)
    parser.add_argument("--model", default=None, help="default: yolo26n.pt, or yolo26n-pose.pt for --mode head")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="override the detection classes --mode would otherwise pick",
    )
    parser.add_argument(
        "--mode",
        choices=["body", "head", "animal"],
        default="body",
        help="body: aim at bbox center, any class. head: aim at head keypoints "
        "(needs a pose model, person only). animal: aim at bbox center, "
        "restricted to COCO bird/cat/dog classes.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print per-frame pixel error / goal position"
    )
    args = parser.parse_args()
    if args.rtsp_pip and not args.rtsp:
        parser.error("--rtsp-pip requires --rtsp")

    target_mode = "head" if args.mode == "head" else "body"
    classes = args.classes if args.classes is not None else (
        list(ANIMAL_CLASS_IDS) if args.mode == "animal" else None
    )

    try:
        wide_handoff_calibration = load_wide_handoff_config(args.wide_handoff_config)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    model_path = args.model or ("yolo26n-pose.pt" if args.mode == "head" else "yolo26n.pt")
    cameras = load_cameras_config(args.camera_config)
    telephoto, wide = cameras.telephoto, cameras.wide

    rtsp_server = None
    if args.rtsp:
        # The RTSP feed's dims are the wide camera's when compositing PIP
        # (it's the background frame), otherwise the tracking camera's.
        rtsp_dims = wide if args.rtsp_pip else telephoto
        rtsp_server = RtspCameraServer(
            width=rtsp_dims.capture_width,
            height=rtsp_dims.capture_height,
            framerate=rtsp_dims.framerate,
            port=args.rtsp_port,
        )
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

    config = load_dynamixel_config(args.servo_config)
    with PanTiltController(config) as controller, ExitStack() as stack:
        telephoto_camera = stack.enter_context(
            GStreamerCameraSource(
                sensor_id=telephoto.sensor_id,
                capture_width=telephoto.capture_width,
                capture_height=telephoto.capture_height,
                framerate=telephoto.framerate,
                flip_method=telephoto.flip_method,
            )
        )
        # Always open, not just for --rtsp-pip: wide is what drives
        # acquisition whenever telephoto has no lock.
        wide_camera = stack.enter_context(
            GStreamerCameraSource(
                sensor_id=wide.sensor_id,
                capture_width=wide.capture_width,
                capture_height=wide.capture_height,
                framerate=wide.framerate,
                flip_method=wide.flip_method,
            )
        )

        relay = PipRtspRelay(rtsp_server, scale=args.pip_scale, margin=args.pip_margin) if rtsp_server else None

        def on_frame(frame) -> None:
            main_frame, inset_frame = frame, None
            if args.rtsp_pip:
                wide_frame = wide_camera.read()
                if wide_frame is not None:
                    main_frame, inset_frame = wide_frame, frame
            draw_mode_badge(main_frame, wide_driven=tele_loop.track_manager.locked_track_id is None)
            relay.push(main_frame, inset_frame)

        controller.initialize()
        telephoto_detector = YoloDetector(model_path=model_path, classes=classes)
        wide_detector = YoloDetector(model_path=model_path, classes=classes)
        wide_track_manager = TrackManager(target_mode=target_mode)
        wide_mapper = WideHandoffMapper(
            wide_handoff_calibration, frame_center=(wide.capture_width / 2, wide.capture_height / 2)
        )

        tele_loop = TrackingLoop(
            camera=telephoto_camera,
            detector=telephoto_detector,
            controller=controller,
            pan_gain=ProportionalGain(kp=PAN_KP, deadband_px=DEADBAND_PX),
            tilt_gain=ProportionalGain(kp=TILT_KP, deadband_px=DEADBAND_PX),
            on_frame=on_frame if rtsp_server is not None else None,
            on_step=on_step if args.verbose else None,
            target_mode=target_mode,
            draw_overlay=args.overlay,
        )

        def run_acquisition_step() -> None:
            """Wide-driven coarse positioning: only called when telephoto
            currently has no lock of its own."""
            wide_frame = wide_camera.read()
            if wide_frame is None:
                return
            detections = wide_detector.track(wide_frame)
            frame_h, frame_w = wide_frame.shape[:2]
            target = wide_track_manager.update(detections, (frame_w / 2, frame_h / 2))
            if target is None:
                return
            x, y = target.target_point(target_mode)
            pan_goal, tilt_goal = wide_mapper.pixel_to_goal(x, y)
            pan_goal = clamp_position(pan_goal, config.pan_limits)
            tilt_goal = clamp_position(tilt_goal, config.tilt_limits)
            controller.sync_write_goal_positions(pan_goal, tilt_goal)
            if args.verbose:
                print(f"[wide handoff] goal=({pan_goal:5d}, {tilt_goal:5d})")

        try:
            while tele_loop.step():
                if tele_loop.track_manager.locked_track_id is None:
                    run_acquisition_step()
        except KeyboardInterrupt:
            logger.info("Stopping")
        finally:
            controller.shutdown()
            if rtsp_server is not None:
                rtsp_server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
