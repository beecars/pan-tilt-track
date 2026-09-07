"""Wires camera -> detector -> target selection -> outer-loop gain ->
sync-write goal positions, one frame at a time."""

from __future__ import annotations

import logging
import time
from typing import Callable

from ..camera.source import CameraSource
from ..dynamixel.controller import PanTiltController
from ..tracking.detector import YoloDetector
from ..tracking.overlay import draw_debug_hud, draw_tracking_overlay
from ..tracking.target import select_target
from .gain import ProportionalGain

logger = logging.getLogger(__name__)


class TrackingLoop:
    def __init__(
        self,
        camera: CameraSource,
        detector: YoloDetector,
        controller: PanTiltController,
        pan_gain: ProportionalGain,
        tilt_gain: ProportionalGain,
        on_step: Callable[[dict], None] | None = None,
        on_frame: Callable[[object], None] | None = None,
        target_mode: str = "body",
        draw_overlay: bool = False,
    ):
        self.camera = camera
        self.detector = detector
        self.controller = controller
        self.pan_gain = pan_gain
        self.tilt_gain = tilt_gain
        # Optional per-frame diagnostic hook, e.g. scripts/sign_check.py.
        self.on_step = on_step
        # Optional raw-frame hook, e.g. RtspCameraServer.push_frame.
        self.on_frame = on_frame
        # "body" (bbox center) or "head" (head keypoint centroid, falls
        # back to body center when keypoints aren't visible).
        self.target_mode = target_mode
        # Burns diagnostics into the frame before on_frame; adds one
        # inference's worth of latency, so off by default.
        self.draw_overlay = draw_overlay
        self._locked_track_id: int | None = None
        # Wall-clock timestamp of the previous step() call, used only to
        # compute the observed loop period for the debug HUD.
        self._last_step_time: float | None = None

    def step(self) -> bool:
        """Process one frame. Returns False if the camera has no frame."""
        now = time.perf_counter()
        frame_interval_ms = (
            (now - self._last_step_time) * 1000 if self._last_step_time is not None else None
        )
        self._last_step_time = now

        frame = self.camera.read()
        if frame is None:
            return False

        if self.on_frame is not None and not self.draw_overlay:
            self.on_frame(frame)

        height, width = frame.shape[:2]
        frame_center = (width / 2, height / 2)

        detections = self.detector.track(frame)
        detect_timing = self.detector.last_timing

        target = select_target(detections, frame_center, self._locked_track_id, self.target_mode)
        self._locked_track_id = target.track_id if target is not None else None

        # pan_position/tilt_position stay None unless a goal is actually
        # written this frame (on_step's "moved" signal). debug_pan_position/
        # debug_tilt_position are HUD-only and also cover the idle case.
        pixel_error_x = pixel_error_y = None
        pan_delta = tilt_delta = None
        pan_position = tilt_position = None
        debug_pan_position = debug_tilt_position = None
        servo_io_ms = None

        if target is not None:
            target_x, target_y = target.target_point(self.target_mode)
            pixel_error_x = target_x - frame_center[0]
            pixel_error_y = target_y - frame_center[1]

            pan_delta = self.pan_gain.compute(pixel_error_x)
            tilt_delta = self.tilt_gain.compute(pixel_error_y)

            cfg = self.controller.config
            if pan_delta != 0 or tilt_delta != 0:
                t0 = time.perf_counter()
                pan_position = self.controller.read_present_position(cfg.pan_id) + pan_delta
                tilt_position = self.controller.read_present_position(cfg.tilt_id) + tilt_delta

                pan_position = _clamp(
                    pan_position, cfg.pan_limits.min_position, cfg.pan_limits.max_position
                )
                tilt_position = _clamp(
                    tilt_position, cfg.tilt_limits.min_position, cfg.tilt_limits.max_position
                )

                self.controller.sync_write_goal_positions(pan_position, tilt_position)
                servo_io_ms = (time.perf_counter() - t0) * 1000
                debug_pan_position, debug_tilt_position = pan_position, tilt_position
            elif self.draw_overlay:
                # In the deadband -- no goal written, but read position for the HUD.
                t0 = time.perf_counter()
                debug_pan_position = self.controller.read_present_position(cfg.pan_id)
                debug_tilt_position = self.controller.read_present_position(cfg.tilt_id)
                servo_io_ms = (time.perf_counter() - t0) * 1000

        if self.draw_overlay:
            t0 = time.perf_counter()
            draw_tracking_overlay(
                frame,
                detections,
                target,
                self.target_mode,
                frame_center,
                self.pan_gain.deadband_px,
                self.tilt_gain.deadband_px,
            )
            draw_ms = (time.perf_counter() - t0) * 1000
            draw_debug_hud(
                frame,
                {
                    "frame_interval_ms": frame_interval_ms,
                    "preprocess_ms": detect_timing.get("preprocess_ms"),
                    "nn_inference_ms": detect_timing.get("nn_inference_ms"),
                    "postprocess_ms": detect_timing.get("postprocess_ms"),
                    "track_ms": detect_timing.get("track_ms"),
                    "detect_total_ms": detect_timing.get("total_ms"),
                    "draw_ms": draw_ms,
                    "servo_io_ms": servo_io_ms,
                    "num_detections": len(detections),
                    "track_id": target.track_id if target is not None else None,
                    "pan_position": debug_pan_position,
                    "tilt_position": debug_tilt_position,
                    "pan_delta": pan_delta,
                    "tilt_delta": tilt_delta,
                },
            )
            if self.on_frame is not None:
                self.on_frame(frame)

        if self.on_step is not None:
            if target is None:
                self.on_step(
                    {
                        "pixel_error_x": None,
                        "pixel_error_y": None,
                        "pan_delta": None,
                        "tilt_delta": None,
                        "pan_position": None,
                        "tilt_position": None,
                        "num_detections": len(detections),
                    }
                )
            elif pan_delta == 0 and tilt_delta == 0:
                self.on_step(
                    {
                        "pixel_error_x": pixel_error_x,
                        "pixel_error_y": pixel_error_y,
                        "pan_delta": 0,
                        "tilt_delta": 0,
                        "pan_position": None,
                        "tilt_position": None,
                    }
                )
            else:
                self.on_step(
                    {
                        "pixel_error_x": pixel_error_x,
                        "pixel_error_y": pixel_error_y,
                        "pan_delta": pan_delta,
                        "tilt_delta": tilt_delta,
                        "pan_position": pan_position,
                        "tilt_position": tilt_position,
                    }
                )
        return True

    def run(self) -> None:
        while self.step():
            pass


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))
