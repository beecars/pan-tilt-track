"""ClipRecorder: writes frames handed to it into a fixed-length MP4 clip
on the local drive, for headless recording (see --record in
scripts/run_tracker.py) where there's no RTSP viewer to capture from."""

from __future__ import annotations

import time
from pathlib import Path

import cv2


class ClipRecorder:
    def __init__(
        self,
        output_dir: Path | str,
        width: int,
        height: int,
        framerate: float,
        max_seconds: float = 10.0,
        fourcc: str = "mp4v",
    ):
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.framerate = framerate
        self.max_frames = max(1, round(max_seconds * framerate))
        self.fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0
        self._path: Path | None = None

    @property
    def active(self) -> bool:
        return self._writer is not None

    @property
    def elapsed_seconds(self) -> float:
        return self._frame_count / self.framerate

    @property
    def total_seconds(self) -> float:
        return self.max_frames / self.framerate

    def start(self) -> Path:
        """Begin a new clip, finalizing any clip already in progress
        first. Returns the output path."""
        if self.active:
            self.stop()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.output_dir / f"clip_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        self._writer = cv2.VideoWriter(
            str(self._path),
            cv2.VideoWriter_fourcc(*self.fourcc),
            self.framerate,
            (self.width, self.height),
        )
        self._frame_count = 0
        return self._path

    def write(self, frame) -> Path | None:
        """No-op unless a clip is in progress. Returns the finalized path
        once the max_seconds cap is reached (auto-stopping this clip),
        otherwise None."""
        if not self.active:
            return None
        self._writer.write(frame)
        self._frame_count += 1
        if self._frame_count >= self.max_frames:
            return self.stop()
        return None

    def stop(self) -> Path | None:
        """Finalize the in-progress clip, if any, and return its path."""
        if not self.active:
            return None
        self._writer.release()
        self._writer = None
        path, self._path = self._path, None
        return path
