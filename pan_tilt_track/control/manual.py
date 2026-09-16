"""Manual pan/tilt override: 'm' toggles it, wasd nudges the servos by a
fixed tick step per keypress while active. TrackingLoop checks
ManualOverride.enabled and skips its own servo writes while it's on, so
this and auto-tracking never fight over the goal position."""

from __future__ import annotations

from ..dynamixel.config import clamp_position
from ..dynamixel.controller import PanTiltController

MANUAL_STEP_TICKS = 20

# wasd -> (pan_delta, tilt_delta) tick steps. Signs are mount-specific
# (see gains.py's PAN_KP note) -- flip here if a/d or w/s move backwards.
KEY_DELTAS = {
    "a": (-MANUAL_STEP_TICKS, 0),
    "d": (MANUAL_STEP_TICKS, 0),
    "w": (0, MANUAL_STEP_TICKS),
    "s": (0, -MANUAL_STEP_TICKS),
}


class ManualOverride:
    def __init__(self) -> None:
        self.enabled = False

    def handle_key(self, key: str | None, controller: PanTiltController) -> None:
        """Consumes one keypress (a no-op if `key` is None)."""
        if key == "m":
            self.enabled = not self.enabled
            print(f"[manual] override {'ON' if self.enabled else 'OFF'}")
        elif self.enabled and key in KEY_DELTAS:
            pan_step, tilt_step = KEY_DELTAS[key]
            cfg = controller.config
            pan = clamp_position(controller.read_present_position(cfg.pan_id) + pan_step, cfg.pan_limits)
            tilt = clamp_position(controller.read_present_position(cfg.tilt_id) + tilt_step, cfg.tilt_limits)
            controller.sync_write_goal_positions(pan, tilt)
