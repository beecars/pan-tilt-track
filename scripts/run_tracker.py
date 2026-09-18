#!/usr/bin/env python3
"""Main entrypoint: two-stage wide-acquisition / telephoto-fine-tracking
pan/tilt correction. Pass --rtsp to also serve the feed for remote
viewing. See pan_tilt_track.control.wide_handoff and TrackingLoop for the
two stages."""

import argparse
import logging
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import cv2

from pan_tilt_track.camera.config import DEFAULT_CAMERA_CONFIG_PATH, load_cameras_config
from pan_tilt_track.camera.gstreamer_source import GStreamerCameraSource
from pan_tilt_track.stream.clip_recorder import ClipRecorder
from pan_tilt_track.stream.pip_compositor import compose_pip
from pan_tilt_track.stream.pip_relay import PipRtspRelay
from pan_tilt_track.stream.rtsp_stream import RtspCameraServer
from pan_tilt_track.control.gain import ProportionalGain
from pan_tilt_track.control.gains import DEADBAND_PX, PAN_KP, TILT_KP
from pan_tilt_track.control.keyboard import NonBlockingKeyReader
from pan_tilt_track.control.loop import TrackingLoop
from pan_tilt_track.control.manual import ManualOverride
from pan_tilt_track.control.wide_handoff import (
    DEFAULT_WIDE_HANDOFF_CONFIG_PATH,
    WideHandoffMapper,
    load_wide_handoff_config,
)
from pan_tilt_track.dynamixel import DEFAULT_SERVO_CONFIG_PATH, PanTiltController, load_dynamixel_config
from pan_tilt_track.dynamixel.config import clamp_position
from pan_tilt_track.tracking.detector import ANIMAL_CLASS_IDS, Detection, YoloDetector
from pan_tilt_track.tracking.overlay import draw_mode_badge
from pan_tilt_track.tracking.track_manager import TrackManager
from pan_tilt_track.ui import LiveDashboard, PlainReporter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CAPTURE_DIR = Path("captures")
CLIP_SECONDS = 10.0
STATE_LABELS = {"MANUAL": "MANUAL", "ACQUIRE": "ACQUIRE (wide)", "HANDOFF": "HANDOFF (tele)"}


def save_capture(frame, reporter) -> None:
    """Writes `frame` -- exactly what's being streamed, badge/overlay/pip
    inset and all -- as a lossless PNG at its native resolution."""
    CAPTURE_DIR.mkdir(exist_ok=True)
    path = CAPTURE_DIR / f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(path), frame)
    reporter.log(f"[capture] saved {path}")


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
    parser.add_argument(
        "--record",
        action="store_true",
        help=f"headless mode: press 'v' to save a {CLIP_SECONDS:.0f}s PIP clip (telephoto large, "
        "wide inset) to disk; works with or without --rtsp",
    )
    parser.add_argument("--clips-dir", default="clips", help="output directory for --record clips")
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
    parser.add_argument(
        "--plain",
        action="store_true",
        help="plain scrolling print() output instead of the live rich dashboard "
        "(automatic when stdout isn't a terminal, e.g. piped/logged runs)",
    )
    parser.add_argument(
        "--det-imgsz",
        type=int,
        default=None,
        help="inference resolution (default: Ultralytics' own default, 640); "
        "raising this trades more nn_inference_ms for more detail on small/distant subjects",
    )
    args = parser.parse_args()
    if args.rtsp_pip and not args.rtsp:
        parser.error("--rtsp-pip requires --rtsp")
    # rich's Live assumes a real terminal; fall back automatically for a
    # piped/logged run even if --plain wasn't passed.
    plain = args.plain or not sys.stdout.isatty()

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

    config = load_dynamixel_config(args.servo_config)
    wide_mapper = WideHandoffMapper(
        wide_handoff_calibration, frame_center=(wide.capture_width / 2, wide.capture_height / 2)
    )
    # The pan/tilt range wide's own FOV can actually command, per its
    # handoff calibration -- shown as a highlighted sub-range on the
    # dashboard's gauges so it's clear how much of the full joint travel
    # wide can see/steer into vs. only reachable by other means (manual).
    wide_pan_bounds = tuple(
        sorted(
            clamp_position(wide_mapper.pixel_to_goal(x, wide.capture_height / 2)[0], config.pan_limits)
            for x in (0, wide.capture_width)
        )
    )
    wide_tilt_bounds = tuple(
        sorted(
            clamp_position(wide_mapper.pixel_to_goal(wide.capture_width / 2, y)[1], config.tilt_limits)
            for y in (0, wide.capture_height)
        )
    )
    reporter = (
        PlainReporter()
        if plain
        else LiveDashboard(
            config.pan_limits,
            config.tilt_limits,
            wide_cam=f"{wide.capture_width}x{wide.capture_height}",
            tele_cam=f"{telephoto.capture_width}x{telephoto.capture_height}",
            wide_pan_bounds=wide_pan_bounds,
            wide_tilt_bounds=wide_tilt_bounds,
        )
    )

    def on_step(info: dict) -> None:
        if info["pixel_error_x"] is None:
            reporter.log(f"no target ({info.get('num_detections', 0)} detections this frame)")
        elif info["pan_position"] is None:
            reporter.log(f"err=({info['pixel_error_x']:+7.1f}, {info['pixel_error_y']:+7.1f})  (in deadband)")
        else:
            reporter.log(
                f"err=({info['pixel_error_x']:+7.1f}, {info['pixel_error_y']:+7.1f})  "
                f"delta=({info['pan_delta']:+5d}, {info['tilt_delta']:+5d})  "
                f"goal=({info['pan_position']:5d}, {info['tilt_position']:5d})"
            )

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
        recorder = (
            ClipRecorder(
                args.clips_dir,
                width=telephoto.capture_width,
                height=telephoto.capture_height,
                framerate=telephoto.framerate,
                max_seconds=CLIP_SECONDS,
            )
            if args.record
            else None
        )
        # Toggled by 'p' below; both cameras capture at the same resolution
        # so either can be the RTSP appsrc's fixed-caps main frame.
        pip_swapped = False
        # Set by 'c' below; consumed by the next on_frame call.
        capture_pending = False

        def on_frame(frame) -> None:
            nonlocal capture_pending
            wide_frame = None
            if args.rtsp_pip or (recorder is not None and recorder.active):
                wide_frame = wide_camera.read()

            main_frame, inset_frame = frame, None
            if args.rtsp_pip and wide_frame is not None:
                main_frame, inset_frame = (
                    (frame, wide_frame) if pip_swapped else (wide_frame, frame)
                )
            draw_mode_badge(main_frame, wide_driven=tele_loop.track_manager.locked_track_id is None)

            if capture_pending:
                capture_pending = False
                # Same compositing relay.push() does below, done once more
                # here so the saved PNG matches the stream exactly -- a
                # capture straight off the frame avoids the RTSP path's
                # lossy H.264 encode.
                composited = (
                    compose_pip(main_frame, inset_frame, scale=args.pip_scale, margin=args.pip_margin)
                    if inset_frame is not None
                    else main_frame
                )
                save_capture(composited, reporter)

            if rtsp_server is not None:
                relay.push(main_frame, inset_frame)

            if recorder is not None and recorder.active:
                # Clips are always telephoto-large / wide-inset, independent
                # of the RTSP pip_swapped toggle above -- when args.rtsp_pip
                # put wide in `main_frame` instead, `frame` (telephoto) still
                # needs its own badge burned in.
                if frame is not main_frame:
                    draw_mode_badge(frame, wide_driven=tele_loop.track_manager.locked_track_id is None)
                clip_frame = (
                    compose_pip(frame, wide_frame, scale=args.pip_scale, margin=args.pip_margin)
                    if wide_frame is not None
                    else frame
                )
                finished_clip = recorder.write(clip_frame)
                if finished_clip is not None:
                    reporter.log(f"[record] saved {finished_clip}")

        controller.initialize()
        detector = YoloDetector(model_path=model_path, classes=classes, imgsz=args.det_imgsz)
        wide_track_manager = TrackManager(target_mode=target_mode)
        manual = ManualOverride()

        tele_loop = TrackingLoop(
            camera=telephoto_camera,
            detector=detector,
            controller=controller,
            pan_gain=ProportionalGain(kp=PAN_KP, deadband_px=DEADBAND_PX),
            tilt_gain=ProportionalGain(kp=TILT_KP, deadband_px=DEADBAND_PX),
            on_frame=on_frame if (rtsp_server is not None or recorder is not None) else None,
            on_step=on_step if args.verbose else None,
            target_mode=target_mode,
            draw_overlay=args.overlay,
            manual_override=manual,
            # Never burn the HUD into the pip inset: skip it whenever
            # telephoto isn't currently the large main frame.
            overlay_active=(lambda: not args.rtsp_pip or pip_swapped),
        )

        current_state: str | None = None
        last_wide_detections = 0
        last_wide_timing: dict[str, float] = {}
        last_detector_source: str | None = None

        def note_state(state: str, num_detections: int) -> None:
            nonlocal current_state
            if state != current_state:
                reporter.log(f"[state] {STATE_LABELS.get(state, state)} ({num_detections} detections)")
                current_state = state

        def run_acquisition_step() -> Detection | None:
            """Wide-driven coarse positioning: only called when telephoto
            currently has no lock of its own. Returns wide's selected
            target (or None), steering toward it first if found."""
            nonlocal last_wide_detections, last_wide_timing, last_detector_source
            wide_frame = wide_camera.read()
            if wide_frame is None:
                return None
            detections = detector.track(wide_frame, reset=last_detector_source != "wide")
            last_detector_source = "wide"
            last_wide_detections = len(detections)
            last_wide_timing = detector.last_timing
            frame_h, frame_w = wide_frame.shape[:2]
            target = wide_track_manager.update(detections, (frame_w / 2, frame_h / 2))
            if target is None:
                if args.verbose:
                    reporter.log(f"[wide handoff] no target ({len(detections)} detections this frame)")
                return None
            x, y = target.target_point(target_mode)
            pan_goal, tilt_goal = wide_mapper.pixel_to_goal(x, y)
            pan_goal = clamp_position(pan_goal, config.pan_limits)
            tilt_goal = clamp_position(tilt_goal, config.tilt_limits)
            controller.sync_write_goal_positions(pan_goal, tilt_goal)
            if args.verbose:
                reporter.log(f"[wide handoff] goal=({pan_goal:5d}, {tilt_goal:5d})")
            return target

        # Entered here, not up in the outer `with` -- camera/model setup
        # above logs plenty of its own (nvarguscamerasrc, Ultralytics
        # weight loading) that must land as normal scrolling output, not
        # get fought over by the Live-managed region below.
        reporter.__enter__()
        reporter.log("Press 'm' to toggle manual pan/tilt override, then wasd to steer.")
        if args.rtsp_pip:
            reporter.log("Press 'p' to swap the RTSP main/inset cameras.")
        if rtsp_server is not None:
            reporter.log(f"Press 'c' to save a full-res PNG of the current stream frame to {CAPTURE_DIR}/.")
        if recorder is not None:
            reporter.log(
                f"Recording a {CLIP_SECONDS:.0f}s PIP clip (telephoto large, wide inset) to "
                f"{args.clips_dir}/ each time telephoto acquires a lock."
            )
        was_locked = False
        try:
            with NonBlockingKeyReader() as keys:
                while True:
                    is_locked = tele_loop.track_manager.locked_track_id is not None

                    if manual.enabled:
                        ok = tele_loop.step(detect=False)
                        state = "MANUAL"
                    elif is_locked:
                        ok = tele_loop.step(detect=True, reset=last_detector_source != "tele")
                        last_detector_source = "tele"
                        state = "HANDOFF"
                    else:
                        wide_target = run_acquisition_step()
                        if wide_target is not None:
                            ok = tele_loop.step(detect=True, reset=last_detector_source != "tele")
                            last_detector_source = "tele"
                            is_locked = tele_loop.track_manager.locked_track_id is not None
                        else:
                            ok = tele_loop.step(detect=False)
                        state = "HANDOFF" if is_locked else "ACQUIRE"
                    if not ok:
                        break

                    key = keys.read_key()
                    manual.handle_key(key, controller)
                    if args.rtsp_pip and key == "p":
                        pip_swapped = not pip_swapped
                        reporter.log(f"[pip] main={'telephoto' if pip_swapped else 'wide'}")
                    if rtsp_server is not None and key == "c":
                        capture_pending = True

                    is_locked = tele_loop.track_manager.locked_track_id is not None
                    if recorder is not None and is_locked and not was_locked and not recorder.active:
                        # not recorder.active: a brief lock flicker (dropped
                        # detection, ByteTrack ID churn) shouldn't truncate
                        # an in-progress clip and restart a new one -- let
                        # it run to its natural 10s end instead.
                        reporter.log(f"[record] lock acquired -- recording {CLIP_SECONDS:.0f}s clip to {recorder.start()}")
                    was_locked = is_locked

                    num_detections = (
                        tele_loop.last_num_detections if state == "HANDOFF" else last_wide_detections
                    )
                    note_state(state, num_detections)

                    pan_ticks = tilt_ticks = None
                    if not plain:
                        pan_ticks = controller.read_present_position(config.pan_id)
                        tilt_ticks = controller.read_present_position(config.tilt_id)
                    reporter.update(
                        state=state,
                        wide_detections=last_wide_detections,
                        tele_detections=tele_loop.last_num_detections,
                        wide_timing=last_wide_timing,
                        tele_timing=tele_loop.last_detect_timing,
                        tele_frame_interval_ms=tele_loop.last_frame_interval_ms,
                        pan_ticks=pan_ticks,
                        tilt_ticks=tilt_ticks,
                    )
        except KeyboardInterrupt:
            reporter.log("Stopping")
        finally:
            controller.shutdown()
            if rtsp_server is not None:
                rtsp_server.stop()
            if recorder is not None and recorder.active:
                finished_clip = recorder.stop()
                reporter.log(f"[record] saved {finished_clip}")
            reporter.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
