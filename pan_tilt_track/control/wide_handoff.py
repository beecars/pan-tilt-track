"""Loads/saves the per-rig wide-camera-pixel -> pan/tilt-goal-tick
calibration (config/wide_handoff.json) and maps a pixel location through
it to an absolute goal position."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_WIDE_HANDOFF_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "wide_handoff.json"


@dataclass(frozen=True)
class WideHandoffCalibration:
    pan_tick_at_center: float
    pan_ticks_per_px: float
    tilt_tick_at_center: float
    tilt_ticks_per_px: float


def load_wide_handoff_config(
    path: Path | str = DEFAULT_WIDE_HANDOFF_CONFIG_PATH,
) -> WideHandoffCalibration:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/calibrate_wide_handoff.py first to produce it."
        )
    data = json.loads(path.read_text())
    try:
        return WideHandoffCalibration(
            pan_tick_at_center=data["pan_tick_at_center"],
            pan_ticks_per_px=data["pan_ticks_per_px"],
            tilt_tick_at_center=data["tilt_tick_at_center"],
            tilt_ticks_per_px=data["tilt_ticks_per_px"],
        )
    except KeyError as e:
        raise ValueError(f"{path}: missing required field {e}") from e


def save_wide_handoff_config(
    calibration: WideHandoffCalibration, path: Path | str = DEFAULT_WIDE_HANDOFF_CONFIG_PATH
) -> None:
    path = Path(path)
    path.write_text(json.dumps(asdict(calibration), indent=2) + "\n")


class WideHandoffMapper:
    def __init__(self, calibration: WideHandoffCalibration, frame_center: tuple[float, float]):
        self.calibration = calibration
        self.frame_center = frame_center

    def pixel_to_goal(self, x: float, y: float) -> tuple[int, int]:
        """Absolute (pan_goal, tilt_goal) tick positions for a target at
        wide-frame pixel (x, y). Caller is responsible for clamping to
        joint limits: this is a raw linear extrapolation and can
        overshoot them near the frame edges."""
        c = self.calibration
        dx = x - self.frame_center[0]
        dy = y - self.frame_center[1]
        pan_goal = round(c.pan_tick_at_center + dx * c.pan_ticks_per_px)
        tilt_goal = round(c.tilt_tick_at_center + dy * c.tilt_ticks_per_px)
        return pan_goal, tilt_goal
