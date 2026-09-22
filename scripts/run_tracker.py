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


def parse_imgsz(value: str) -> int | list[int]:
    """--det-imgsz value: a bare int for square inference, or "H,W" for
    rect inference at a non-square aspect (avoids the wasted padding a
    square imgsz forces on a 16:9 frame). Only meaningful for a .pt model
    -- YoloDetector ignores imgsz entirely for exported formats, whose
    shape is fixed at export time."""
    if "," in value:
        parts = value.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"--det-imgsz rect form must be 'H,W', got {value!r}")
        try:
            return [int(p) for p in parts]
        except ValueError:
            raise argparse.ArgumentTypeError(f"--det-imgsz rect form must be 'H,W' integers, got {value!r}") from None
    return int(value)


def downsample_to(frame, size: tuple[int, int]):
    """Resizes `frame` to `size` (w, h) if it isn't already, for feeding the
    detector/RTSP at the wide-handoff calibration's resolution regardless of
    the camera's actual capture resolution."""
    if frame is None or (frame.shape[1], frame.shape[0]) == size:
        return frame
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def save_capture(frame, reporter, prefix: str = "capture") -> Path:
    """Writes the raw wide-camera `frame` (no overlay) as a lossless PNG."""
    CAPTURE_DIR.mkdir(exist_ok=True)
    ms = int((time.time() % 1) * 1000)
    path = CAPTURE_DIR / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{ms:03d}.png"
    cv2.imwrite(str(path), frame)
    reporter.log(f"[capture] saved {path}")
    return path


def save_yolo_labels(detections, frame_size: tuple[int, int], png_path: Path) -> None:
    """Writes a YOLO-format .txt label file next to `png_path` (same stem):
    one "class_id x_center y_center width height" line per detection,
    normalized 0-1 against `frame_size` (the frame the detector actually
    ran on). Normalized coordinates carry over unchanged to `png_path`
    even when it's a different (e.g. native 4K) resolution, since resizing
    scales each axis by a uniform fraction."""
    frame_w, frame_h = frame_size
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det.box
        xc = (x1 + x2) / 2 / frame_w
        yc = (y1 + y2) / 2 / frame_h
        w = (x2 - x1) / frame_w
        h = (y2 - y1) / frame_h
        lines.append(f"{det.class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    png_path.with_suffix(".txt").write_text("\n".join(lines) + ("\n" if lines else ""))


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
    parser.add_argument(
        "--wide-capture-width",
        type=int,
        default=None,
        help="override the wide camera's Argus sensor-mode width (e.g. 3840 for 4K); detection runs "
        "on the native frame (see --det-imgsz), only the RTSP/PIP inset is downsampled to the "
        "camera-config resolution",
    )
    parser.add_argument(
        "--wide-capture-height",
        type=int,
        default=None,
        help="override the wide camera's Argus sensor-mode height (e.g. 2160 for 4K)",
    )
    parser.add_argument(
        "--wide-capture-framerate",
        type=int,
        default=None,
        help="override the wide camera's Argus sensor-mode framerate (sensor modes pair fixed "
        "resolutions with fixed framerates, e.g. this sensor's 4K mode is ~30fps vs 1080p's ~60fps)",
    )
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
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=None,
        help=f"auto-save a raw wide-camera PNG to {CAPTURE_DIR}/ every N seconds, regardless of detections "
        "(dataset collection)",
    )
    parser.add_argument(
        "--capture-on-detect-interval",
        type=float,
        default=None,
        help=f"auto-save a raw wide-camera PNG to {CAPTURE_DIR}/ at most every N seconds while the wide "
        "camera has a detection, alongside a same-named YOLO-format .txt label file (dataset collection)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model for both cameras (default: yolo26n.pt, or yolo26n-pose.pt for --mode head); "
        "--wide-model/--tele-model override it per camera",
    )
    parser.add_argument(
        "--wide-model",
        default=None,
        help="wide camera's model, overriding --model. Each camera gets its own detector, so the "
        "two can be engines exported at different resolutions (see scripts/build_engines.py) -- "
        "e.g. yolo26n_1440x2560.engine here for pixels on small/distant subjects",
    )
    parser.add_argument(
        "--tele-model",
        default=None,
        help="telephoto camera's model, overriding --model. Telephoto paces the control loop, so a "
        "smaller engine (e.g. yolo26n_544x960.engine) buys loop rate on an already-zoomed view",
    )
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
        "--wide-only",
        action="store_true",
        help="never hand off to the telephoto camera; stay in wide-driven ACQUIRE steering "
        "continuously (telephoto still opens/streams for --rtsp/--record, just never steers)",
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
        type=parse_imgsz,
        default=None,
        help="inference resolution: a bare int for square (default: Ultralytics' own default, 640), "
        "or 'H,W' for rect (e.g. 2176,3840 to match a 16:9 4K frame without wasteful square padding). "
        "Raising this trades more nn_inference_ms for more detail on small/distant subjects; "
        "ignored entirely for exported (.engine) models, whose shape is fixed at export time",
    )
    parser.add_argument("--wide-det-imgsz", type=parse_imgsz, default=None, help="--det-imgsz for the wide camera only")
    parser.add_argument("--tele-det-imgsz", type=parse_imgsz, default=None, help="--det-imgsz for the telephoto camera only")
    parser.add_argument(
        "--det-conf",
        type=float,
        default=None,
        help="detection confidence threshold (default: 0.1, set by Ultralytics' model.track() "
        "to match bytetrack.yaml's track_low_thresh so ByteTrack gets low-confidence boxes for "
        "its second-stage association)",
    )
    parser.add_argument("--wide-det-conf", type=float, default=None, help="--det-conf for the wide camera only")
    parser.add_argument("--tele-det-conf", type=float, default=None, help="--det-conf for the telephoto camera only")
    args = parser.parse_args()
    if args.rtsp_pip and not args.rtsp:
        parser.error("--rtsp-pip requires --rtsp")
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

    default_model = args.model or ("yolo26n-pose.pt" if args.mode == "head" else "yolo26n.pt")
    wide_model = args.wide_model or default_model
    tele_model = args.tele_model or default_model
    wide_imgsz = args.wide_det_imgsz if args.wide_det_imgsz is not None else args.det_imgsz
    tele_imgsz = args.tele_det_imgsz if args.tele_det_imgsz is not None else args.det_imgsz
    wide_conf = args.wide_det_conf if args.wide_det_conf is not None else args.det_conf
    tele_conf = args.tele_det_conf if args.tele_det_conf is not None else args.det_conf
    cameras = load_cameras_config(args.camera_config)
    telephoto, wide = cameras.telephoto, cameras.wide
    wide_processing_size = (wide.capture_width, wide.capture_height)
    wide_capture_size = (
        args.wide_capture_width or wide.capture_width,
        args.wide_capture_height or wide.capture_height,
    )
    wide_cam_label = f"{wide_capture_size[0]}x{wide_capture_size[1]}"

    rtsp_server = None
    if args.rtsp:
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

    def model_summary(model: str, imgsz, conf) -> str:
        parts = [Path(model).name]
        if imgsz is not None:
            parts.append(f"@{imgsz}" if isinstance(imgsz, int) else "@" + "x".join(str(v) for v in imgsz))
        if conf is not None:
            parts.append(f" conf={conf}")
        return "".join(parts)

    run_config = {
        "mode": args.mode,
        "wide": model_summary(wide_model, wide_imgsz, wide_conf),
        "tele": model_summary(tele_model, tele_imgsz, tele_conf),
        "overlay": "on" if args.overlay else "off",
        "handoff": "off" if args.wide_only else "on",
    }
    reporter = (
        PlainReporter()
        if plain
        else LiveDashboard(
            config.pan_limits,
            config.tilt_limits,
            run_config=run_config,
            wide_cam=wide_cam_label,
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
        wide_camera = stack.enter_context(
            GStreamerCameraSource(
                sensor_id=wide.sensor_id,
                capture_width=wide_capture_size[0],
                capture_height=wide_capture_size[1],
                framerate=args.wide_capture_framerate or wide.framerate,
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
        pip_swapped = False

        def on_frame(frame) -> None:
            wide_frame = None
            if args.rtsp_pip or (recorder is not None and recorder.active):
                wide_frame = downsample_to(wide_camera.read(), wide_processing_size)

            main_frame, inset_frame = frame, None
            if args.rtsp_pip and wide_frame is not None:
                main_frame, inset_frame = (
                    (frame, wide_frame) if pip_swapped else (wide_frame, frame)
                )
            draw_mode_badge(main_frame, wide_driven=tele_loop.track_manager.locked_track_id is None)

            if rtsp_server is not None:
                relay.push(main_frame, inset_frame)

            if recorder is not None and recorder.active:
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
        wide_detector = YoloDetector(model_path=wide_model, classes=classes, imgsz=wide_imgsz, conf=wide_conf)
        tele_detector = YoloDetector(model_path=tele_model, classes=classes, imgsz=tele_imgsz, conf=tele_conf)
        wide_track_manager = TrackManager(target_mode=target_mode)
        manual = ManualOverride()

        tele_loop = TrackingLoop(
            camera=telephoto_camera,
            detector=tele_detector,
            controller=controller,
            pan_gain=ProportionalGain(kp=PAN_KP, deadband_px=DEADBAND_PX),
            tilt_gain=ProportionalGain(kp=TILT_KP, deadband_px=DEADBAND_PX),
            on_frame=on_frame if (rtsp_server is not None or recorder is not None) else None,
            on_step=on_step if args.verbose else None,
            target_mode=target_mode,
            draw_overlay=args.overlay,
            manual_override=manual,
            overlay_active=(lambda: not args.rtsp_pip or pip_swapped),
        )

        current_state: str | None = None
        wide_timing: dict[str, float] = {}
        wide_detections = 0
        wide_needs_reset = False
        last_wide_detections: list[Detection] = []
        last_wide_frame_size = wide_capture_size

        def note_state(state: str, num_detections: int) -> None:
            nonlocal current_state
            if state != current_state:
                reporter.log(f"[state] {STATE_LABELS.get(state, state)} ({num_detections} detections)")
                current_state = state

        def run_acquisition_step() -> Detection | None:
            """Wide-driven coarse positioning: only called when telephoto
            currently has no lock of its own. Returns wide's selected
            target (or None), steering toward it first if found."""
            nonlocal wide_timing, wide_detections, wide_needs_reset, last_wide_detections, last_wide_frame_size
            wide_frame = wide_camera.read()
            if wide_frame is None:
                return None
            detections = wide_detector.track(wide_frame, reset=wide_needs_reset)
            wide_needs_reset = False
            wide_detections = len(detections)
            last_wide_detections = detections
            frame_h, frame_w = wide_frame.shape[:2]
            last_wide_frame_size = (frame_w, frame_h)
            wide_timing = wide_detector.last_timing
            target = wide_track_manager.update(detections, (frame_w / 2, frame_h / 2))
            if target is None:
                if args.verbose:
                    reporter.log(f"[wide handoff] no target ({len(detections)} detections this frame)")
                return None
            x, y = target.target_point(target_mode)
            x *= wide_processing_size[0] / frame_w
            y *= wide_processing_size[1] / frame_h
            pan_goal, tilt_goal = wide_mapper.pixel_to_goal(x, y)
            pan_goal = clamp_position(pan_goal, config.pan_limits)
            tilt_goal = clamp_position(tilt_goal, config.tilt_limits)
            controller.sync_write_goal_positions(pan_goal, tilt_goal)
            if args.verbose:
                reporter.log(f"[wide handoff] goal=({pan_goal:5d}, {tilt_goal:5d})")
            return target

        reporter.__enter__()
        reporter.log("Press 'm' to toggle manual pan/tilt override, then wasd to steer.")
        if args.rtsp_pip:
            reporter.log("Press 'p' to swap the RTSP main/inset cameras.")
        reporter.log(f"Press 'c' to save a full-res wide-camera PNG (for later annotation) to {CAPTURE_DIR}/.")
        if wide_capture_size != wide_processing_size:
            reporter.log(
                f"[wide] capturing at {wide_capture_size[0]}x{wide_capture_size[1]}; detection runs at "
                "native resolution (target coords rescaled to "
                f"{wide_processing_size[0]}x{wide_processing_size[1]} for wide_handoff); the RTSP/PIP "
                f"inset is downsampled to {wide_processing_size[0]}x{wide_processing_size[1]}."
            )
        if args.capture_interval:
            reporter.log(f"Auto-saving a wide-camera PNG to {CAPTURE_DIR}/ every {args.capture_interval:.0f}s.")
        if args.capture_on_detect_interval:
            reporter.log(
                f"Auto-saving a wide-camera PNG + YOLO .txt labels to {CAPTURE_DIR}/ (max every "
                f"{args.capture_on_detect_interval:.0f}s) while the wide camera has a detection."
            )
        if recorder is not None:
            reporter.log(
                f"Recording a {CLIP_SECONDS:.0f}s PIP clip (telephoto large, wide inset) to "
                f"{args.clips_dir}/ each time telephoto acquires a lock."
            )
        was_locked = False
        last_interval_capture = 0.0
        last_detect_capture = 0.0
        try:
            with NonBlockingKeyReader() as keys:
                while True:
                    is_locked = tele_loop.track_manager.locked_track_id is not None
                    wide_ran = tele_ran = False

                    if manual.enabled:
                        ok = tele_loop.step(detect=False)
                        state = "MANUAL"
                    elif is_locked:
                        ok = tele_loop.step(detect=True)
                        tele_ran = True
                        state = "HANDOFF"
                    else:
                        wide_target = run_acquisition_step()
                        wide_ran = True
                        if wide_target is not None and not args.wide_only:
                            ok = tele_loop.step(detect=True)
                            tele_ran = True
                            is_locked = tele_loop.track_manager.locked_track_id is not None
                        else:
                            ok = tele_loop.step(detect=False)
                        state = "HANDOFF" if is_locked else "ACQUIRE"
                    if not ok:
                        break

                    if not wide_ran:
                        wide_timing = {}
                        wide_detections = 0
                        wide_needs_reset = True

                    key = keys.read_key()
                    manual.handle_key(key, controller)
                    if args.rtsp_pip and key == "p":
                        pip_swapped = not pip_swapped
                        reporter.log(f"[pip] main={'telephoto' if pip_swapped else 'wide'}")
                    if key == "c":
                        capture_frame = wide_camera.read()
                        if capture_frame is not None:
                            save_capture(capture_frame, reporter)

                    is_locked = tele_loop.track_manager.locked_track_id is not None
                    if recorder is not None and is_locked and not was_locked and not recorder.active:
                        measured_fps = (
                            1000 / tele_loop.last_frame_interval_ms
                            if tele_loop.last_frame_interval_ms
                            else telephoto.framerate
                        )
                        clip_path = recorder.start(framerate=measured_fps)
                        reporter.log(f"[record] lock acquired -- recording {CLIP_SECONDS:.0f}s clip to {clip_path}")
                    was_locked = is_locked

                    num_detections = (
                        tele_loop.last_num_detections if state == "HANDOFF" else wide_detections
                    )
                    note_state(state, num_detections)

                    now = time.monotonic()
                    if args.capture_interval and now - last_interval_capture >= args.capture_interval:
                        capture_frame = wide_camera.read()
                        if capture_frame is not None:
                            save_capture(capture_frame, reporter, prefix="auto_interval")
                        last_interval_capture = now
                    if (
                        args.capture_on_detect_interval
                        and wide_detections > 0
                        and now - last_detect_capture >= args.capture_on_detect_interval
                    ):
                        capture_frame = wide_camera.read()
                        if capture_frame is not None:
                            png_path = save_capture(capture_frame, reporter, prefix="auto_detect")
                            save_yolo_labels(last_wide_detections, last_wide_frame_size, png_path)
                        last_detect_capture = now

                    pan_ticks = tilt_ticks = None
                    if not plain:
                        pan_ticks = controller.read_present_position(config.pan_id)
                        tilt_ticks = controller.read_present_position(config.tilt_id)
                    locked_target = tele_loop.last_target if state == "HANDOFF" else None
                    reporter.update(
                        state=state,
                        wide_active=wide_ran,
                        tele_active=tele_ran,
                        wide_timing=wide_timing,
                        wide_detections=wide_detections,
                        tele_timing=tele_loop.last_detect_timing,
                        tele_detections=tele_loop.last_num_detections,
                        loop_interval_ms=tele_loop.last_frame_interval_ms,
                        locked_track_id=locked_target.track_id if locked_target is not None else None,
                        locked_confidence=locked_target.confidence if locked_target is not None else None,
                        pan_ticks=pan_ticks,
                        tilt_ticks=tilt_ticks,
                        recording=recorder is not None and recorder.active,
                        recording_elapsed=recorder.elapsed_seconds if recorder is not None and recorder.active else None,
                        recording_total=recorder.total_seconds if recorder is not None and recorder.active else None,
                        rtsp_active=rtsp_server is not None,
                        pip_active=args.rtsp_pip,
                        rtsp_clients=rtsp_server.stats()["clients"] if rtsp_server is not None else 0,
                        rtsp_fps=rtsp_server.stats()["fps"] if rtsp_server is not None else None,
                        clips_saved=recorder.clips_saved if recorder is not None else 0,
                        pip_compose_ms=relay.last_compose_ms if relay is not None else None,
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
