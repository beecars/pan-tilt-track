"""ProportionalGain: computes a goal-position tick delta from pixel error,
with a deadband."""

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
