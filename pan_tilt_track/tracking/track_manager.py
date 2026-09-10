"""TrackManager: maintains a sticky lock on one detection's track ID
across frames."""

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
