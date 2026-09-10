"""Owns track identity/lifecycle state across frames -- split out from
TrackingLoop and target.py's stateless selection policy so it has a home
to grow into.

Today this only remembers which track_id is "locked" (moved here verbatim
from TrackingLoop._locked_track_id) so one camera's per-frame detections
resolve to a single sticky target. This is the intended seam for two
pieces of not-yet-written functionality, noted here so future work lands
in the right place instead of getting grafted onto TrackingLoop or
target.py:

- ID/ReID: re-acquiring a lost track_id via appearance matching instead
  of dropping lock the instant `track_id` disappears from one frame's
  detections (ByteTrack's own IDs aren't stable across an occlusion).
- Multi-camera / world-coordinate fusion: turning per-camera pixel-space
  tracks into world-coordinate tracks for 3D reconstruction. That's a
  distinct capability (needs camera calibration/pose, operates across
  cameras) and will likely be a separate class that composes one
  TrackManager per camera rather than a rewrite of this one -- this
  class intentionally knows nothing about pixel-to-world geometry so it
  doesn't have to be reworked when that lands.

Neither exists yet. Don't add speculative ReID model loading or
world-coordinate math here until there's a second camera or a 3D
pipeline actually consuming it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Detection
from .target import select_target


@dataclass
class TrackManager:
    target_mode: str = "body"
    _locked_track_id: int | None = field(default=None, init=False, repr=False)

    def update(
        self, detections: list[Detection], frame_center: tuple[float, float]
    ) -> Detection | None:
        """Pick this frame's target and update the sticky lock. See
        target.select_target for the selection policy itself."""
        target = select_target(detections, frame_center, self._locked_track_id, self.target_mode)
        self._locked_track_id = target.track_id if target is not None else None
        return target

    @property
    def locked_track_id(self) -> int | None:
        return self._locked_track_id
