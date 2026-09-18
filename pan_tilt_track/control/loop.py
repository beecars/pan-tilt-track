"""Wires camera -> detector -> target selection -> outer-loop gain ->
sync-write goal positions, one frame at a time."""

from __future__ import annotations

import logging
import time
from typing import Callable

from ..camera.source import CameraSource
from ..dynamixel.config import clamp_position
from ..dynamixel.controller import PanTiltController
from ..tracking.detector import YoloDetector
from ..tracking.overlay import draw_debug_hud, draw_tracking_overlay
from ..tracking.track_manager import TrackManager
from .gain import ProportionalGain
from .manual import ManualOverride

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
        track_manager: TrackManager | None = None,
        manual_override: ManualOverride | None = None,
        overlay_active: Callable[[], bool] | None = None,
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
        # Track identity/lifecycle (sticky lock today; see TrackManager's
        # docstring for the ID/ReID and multi-camera fusion it's the seam
        # for). Injectable so a future ReID-capable or multi-camera-aware
        # manager can be swapped in without touching TrackingLoop again.
        self.track_manager = track_manager or TrackManager(target_mode=target_mode)
        # When set and .enabled, wasd is driving the servos directly (see
        # ManualOverride) -- step() still tracks/displays but skips its own
        # servo writes so the two don't fight over the goal position.
        self.manual_override = manual_override
        # Called (when set) to decide whether this frame's overlay/HUD
        # should actually be burned in, e.g. a PIP viewer that only wants
        # them on whichever camera is currently the large main frame --
        # ignored unless draw_overlay is also True.
        self.overlay_active = overlay_active
        # Wall-clock timestamp of the previous step() call, used only to
        # compute the observed loop period for the debug HUD.
        self._last_step_time: float | None = None
        # Detection count from the most recent step(), for callers doing
        # their own state/handoff logging (e.g. scripts/run_tracker.py).
        self.last_num_detections: int = 0
        # Observed loop period from the most recent step(), for callers
        # showing their own FPS/pipeline stats (e.g. LiveDashboard).
        self.last_frame_interval_ms: float | None = None
        # detector.last_timing as of the most recent step() that actually
        # ran detection, for callers reading it after a detector shared
        # with another camera source has since moved on (e.g. LiveDashboard).
        self.last_detect_timing: dict[str, float] = {}

    def step(self, detect: bool = True, reset: bool = False) -> bool:
        """Process one frame. Returns False if the camera has no frame.
        `detect=False` reads the frame and still drives on_frame/overlay,
        but skips the detector call entirely -- for callers that want this
        camera's video kept flowing without paying for inference on it.
        `reset` is forwarded to the detector; see YoloDetector.track."""
        now = time.perf_counter()
        frame_interval_ms = (
            (now - self._last_step_time) * 1000 if self._last_step_time is not None else None
        )
        self._last_step_time = now
        self.last_frame_interval_ms = frame_interval_ms

        frame = self.camera.read()
        if frame is None:
            return False

        should_draw_overlay = self.draw_overlay and (
            self.overlay_active is None or self.overlay_active()
        )

        if self.on_frame is not None and not should_draw_overlay:
            self.on_frame(frame)

        height, width = frame.shape[:2]
        frame_center = (width / 2, height / 2)

        if detect:
            detections = self.detector.track(frame, reset=reset)
            detect_timing = self.detector.last_timing
            self.last_num_detections = len(detections)
            self.last_detect_timing = detect_timing
            target = self.track_manager.update(detections, frame_center)
        else:
            detections = []
            detect_timing = {}
            target = None

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
            if self.manual_override is not None and self.manual_override.enabled:
                pan_delta = tilt_delta = 0

            cfg = self.controller.config
            if pan_delta != 0 or tilt_delta != 0:
                t0 = time.perf_counter()
                pan_position = self.controller.read_present_position(cfg.pan_id) + pan_delta
                tilt_position = self.controller.read_present_position(cfg.tilt_id) + tilt_delta

                pan_position = clamp_position(pan_position, cfg.pan_limits)
                tilt_position = clamp_position(tilt_position, cfg.tilt_limits)

                self.controller.sync_write_goal_positions(pan_position, tilt_position)
                servo_io_ms = (time.perf_counter() - t0) * 1000
                debug_pan_position, debug_tilt_position = pan_position, tilt_position
            elif should_draw_overlay:
                # In the deadband: no goal written, but read position for the HUD.
                t0 = time.perf_counter()
                debug_pan_position = self.controller.read_present_position(cfg.pan_id)
                debug_tilt_position = self.controller.read_present_position(cfg.tilt_id)
                servo_io_ms = (time.perf_counter() - t0) * 1000

        if should_draw_overlay:
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

        if self.on_step is not None and detect:
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
