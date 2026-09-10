"""ProportionalGain: computes a goal-position tick delta from pixel error,
with a deadband."""

from dataclasses import dataclass


@dataclass
class ProportionalGain:
    # No integral/derivative term: the XL330's own firmware position PID
    # and Profile Velocity/Acceleration already smooth a goal-position
    # write into motion.
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
