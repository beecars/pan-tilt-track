"""Outer (vision) loop: pixel error -> servo goal-position delta.

The XL330's own firmware PID (position control mode) and Profile
Velocity/Acceleration already turn a goal-position write into smooth
motion, so this module deliberately does NOT reimplement PID -- it only
decides *what* goal position to ask for, via a proportional gain with a
deadband to avoid hunting near the servo's angular resolution floor.
"""

from dataclasses import dataclass


@dataclass
class ProportionalGain:
    kp: float
    deadband_px: float

    def compute(self, pixel_error: float) -> int:
        """Return a goal-position tick delta for the given pixel error.

        Returns 0 inside the deadband so the head doesn't hunt on noise
        smaller than the servo's angular resolution can usefully act on.
        """
        if abs(pixel_error) < self.deadband_px:
            return 0
        return round(self.kp * pixel_error)
