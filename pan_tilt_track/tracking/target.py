"""Target-selection policy, kept separate from the detector so it can evolve
independently of the underlying model (e.g. add re-identification later)."""

from __future__ import annotations

from .detector import Detection


def select_target(
    detections: list[Detection],
    frame_center: tuple[float, float],
    locked_track_id: int | None = None,
    mode: str = "body",
) -> Detection | None:
    """Pick the detection to track this frame.

    If a previously-locked track_id is still present, stick with it (avoids
    flickering between multiple candidates). Otherwise pick whichever
    detection's target_point(mode) is closest to the frame center.
    """
    if not detections:
        return None

    if locked_track_id is not None:
        for d in detections:
            if d.track_id == locked_track_id:
                return d

    cx, cy = frame_center

    def dist_sq(d: Detection) -> float:
        dx, dy = d.target_point(mode)
        return (dx - cx) ** 2 + (dy - cy) ** 2

    return min(detections, key=dist_sq)
