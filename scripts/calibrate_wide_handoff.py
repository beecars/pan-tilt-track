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
import time

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

MIN_SPREAD_FRACTION = 0.3
RENDER_INTERVAL_S = 0.1

GRID_COLS = 31
GRID_ROWS = 13


class LiveDisplay:
    """Redraws a fixed status block in place (servo/centering/wide state,
    sample counter, and an ASCII map of where in the wide frame samples
    have landed), instead of scrolling one line per event. Requires a
    real terminal -- run this script directly (not piped/redirected).
    """

    def __init__(self, wide_w: float, wide_h: float, min_samples: int, deadband_px: float) -> None:
        self.wide_w = wide_w
        self.wide_h = wide_h
        self.min_samples = min_samples
        self.deadband_px = deadband_px
        self.grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        self._prev_height = 0
        self._last_render = 0.0

    def record_sample(self, dx: float, dy: float) -> None:
        col = int((dx + self.wide_w / 2) / self.wide_w * (GRID_COLS - 1))
        row = int((dy + self.wide_h / 2) / self.wide_h * (GRID_ROWS - 1))
        col = min(max(col, 0), GRID_COLS - 1)
        row = min(max(row, 0), GRID_ROWS - 1)
        self.grid[row][col] += 1

    def _render_map(self) -> list[str]:
        center_row, center_col = GRID_ROWS // 2, GRID_COLS // 2
        lines = []
        for r in range(GRID_ROWS):
            chars = []
            for c in range(GRID_COLS):
                count = self.grid[r][c]
                if count >= 3:
                    chars.append("#")
                elif count >= 1:
                    chars.append("o")
                elif r == center_row and c == center_col:
                    chars.append("+")
                elif r == center_row:
                    chars.append("-")
                elif c == center_col:
                    chars.append("|")
                else:
                    chars.append(".")
            lines.append("  " + "".join(chars))
        return lines

    @staticmethod
    def _bar(value: float, extent: float, width: int = 21) -> str:
        v = max(-extent, min(extent, value))
        pos = round((v + extent) / (2 * extent) * (width - 1))
        chars = ["-"] * width
        chars[width // 2] = "|"
        chars[pos] = "#"
        return "".join(chars)

    def render(
        self,
        num_samples: int,
        servo_status: str,
        pixel_error_x: float | None,
        pixel_error_y: float | None,
        pan_delta: int | None,
        tilt_delta: int | None,
        wide_status: str,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and (now - self._last_render) < RENDER_INTERVAL_S:
            return
        self._last_render = now

        lines = ["Wide-handoff calibration -- Ctrl+C to stop early", ""]

        if pan_delta or tilt_delta:
            lines.append(f"servo:    MOVING   pan={pan_delta:+d}  tilt={tilt_delta:+d}")
        else:
            lines.append(f"servo:    {servo_status}")

        if pixel_error_x is None:
            lines.append("center:   -- (no telephoto lock)")
        else:
            lines.append(
                f"center:   pan  {self._bar(pixel_error_x, self.wide_w / 2)}  {pixel_error_x:+7.1f}px "
                f"(deadband +/-{self.deadband_px:.0f}px)"
            )
            lines.append(
                f"          tilt {self._bar(pixel_error_y, self.wide_h / 2)}  {pixel_error_y:+7.1f}px"
            )

        lines.append(f"wide:     {wide_status}")
        filled = int(num_samples / self.min_samples * 20) if self.min_samples else 0
        filled = min(filled, 20)
        lines.append(f"samples:  {num_samples}/{self.min_samples}  [{'=' * filled}{'-' * (20 - filled)}]")
        lines.append("")
        lines.append("coverage (wide frame):")
        lines.extend(self._render_map())

        out = []
        if self._prev_height:
            out.append(f"\033[{self._prev_height}A")
        for line in lines:
            out.append("\033[2K" + line + "\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self._prev_height = len(lines)


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
    parser.add_argument("--wide-model", default=None, help="wide camera's model, overriding --model")
    parser.add_argument("--tele-model", default=None, help="telephoto camera's model, overriding --model")
    parser.add_argument("--camera-config", default=DEFAULT_CAMERA_CONFIG_PATH)
    parser.add_argument("--servo-config", default=DEFAULT_SERVO_CONFIG_PATH)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--out", default=DEFAULT_WIDE_HANDOFF_CONFIG_PATH)
    args = parser.parse_args()
    default_model = args.model or ("yolo26n-pose.pt" if args.target_mode == "head" else "yolo26n.pt")
    wide_model = args.wide_model or default_model
    tele_model = args.tele_model or default_model

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
        wide_detector = YoloDetector(model_path=wide_model, classes=[0])  # class 0 = person in COCO
        display = LiveDisplay(
            wide_w=wide.capture_width, wide_h=wide.capture_height,
            min_samples=args.min_samples, deadband_px=DEADBAND_PX,
        )

        def on_step(info: dict) -> None:
            pixel_error_x = info.get("pixel_error_x")
            pixel_error_y = info.get("pixel_error_y")
            pan_delta = info.get("pan_delta")
            tilt_delta = info.get("tilt_delta")

            if pixel_error_x is None:
                display.render(len(pan_samples), "SETTLED", None, None, None, None, "-- (no telephoto lock)")
                return  # no target
            if pan_delta or tilt_delta:
                display.render(len(pan_samples), "SETTLED", pixel_error_x, pixel_error_y, pan_delta, tilt_delta, "-- (centering)")
                return  # mid-correction: not well-centered yet

            wide_frame = wide_camera.read()
            if wide_frame is None:
                display.render(len(pan_samples), "SETTLED", pixel_error_x, pixel_error_y, 0, 0, "no frame")
                return
            detections = wide_detector.track(wide_frame)
            if len(detections) != 1:
                display.render(
                    len(pan_samples), "SETTLED", pixel_error_x, pixel_error_y, 0, 0,
                    f"{len(detections)} detections (need exactly 1)",
                )
                return
            wx, wy = detections[0].target_point(args.target_mode)
            wide_h, wide_w = wide_frame.shape[:2]
            dx, dy = wx - wide_w / 2, wy - wide_h / 2
            pan_tick = controller.read_present_position(config.pan_id)
            tilt_tick = controller.read_present_position(config.tilt_id)
            pan_samples.append((dx, pan_tick))
            tilt_samples.append((dy, tilt_tick))
            display.record_sample(dx, dy)
            display.render(
                len(pan_samples), "SETTLED", pixel_error_x, pixel_error_y, 0, 0, "1 detection -> sample recorded",
                force=True,
            )

        telephoto_detector = YoloDetector(model_path=tele_model, classes=[0])
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
